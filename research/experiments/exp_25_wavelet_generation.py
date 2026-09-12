"""Experiment 25 -- generate images from the wavelet HMT and compare the families.

Three transition-kernel families, one inference engine, one dataset:

    gaussian       Gaussian tree. The second-order closure the project uses as a
                   control everywhere else.
    mixture        `MixtureInnovationKernel`: linear-AR, flexible innovation
                   *shape*. Can match heavy-tailed marginals; its conditional
                   variance does not depend on the parent, so it cannot represent
                   cross-scale magnitude dependence.
    scale_mixture  `ScaleMixtureKernel`: conditional *scale* depends on the
                   parent. The family `exp_23` says the data needs.

What is compared, and why in this order (see IMAGE_EXTENSION_STAGE1.md §8):

1. **Held-out exact log-likelihood.** Computable exactly here and not at all for
   a GAN-style baseline. It measures the model, not the sampler.
2. **Per-subband excess kurtosis** of generated images against *held-out real*
   ones -- the marginal shape at every scale.
3. **Cross-scale magnitude excess** -- the statistic that separates a
   hierarchical model from a factorised heavy-tailed one, read against the
   linear-AR null rather than against 1.

FID/KID are deliberately absent: they need `torch`, which is not installed here
or on the cluster, and at this model scale they would largely measure wavelet
reconstruction quality rather than score quality. They belong in the writeup for
comparability, not in the lead.

Parts
-----
fit        Fit each family; record EM traces and held-out likelihood.
generate   Ancestral samples from each family; subband and cross-scale profiles
           against held-out real images.
sampler    Reverse diffusion driven by the exact BP score, against ancestral
           samples of the same model. Isolates sampler error from model error.
grid       PNG contact sheets of real and generated images.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    PACKAGE_ROOT, apply_overrides, experiment_parser, provenance, select_parts,
)
from src.image_data import load_cifar10_luminance
from src.kernels import GaussianAR1Kernel, MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel
from src.utils import ensure_dir, rng_for, write_csv, write_json
from src.wavelet_model import fit_wavelet_tree
from src.wavelet_stats import crossscale_profile, profile_gap, subband_profile

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    "levels": 5,
    "n_train": 1000,
    "n_test": 1000,
    "t_train": (0.4, 0.9),
    "t_likelihood": 0.05,
    "n_iters": 15,
    "t_resolve": 0.05,
    "half_width": 8.0,
    "n_components": 4,
    "families": ("gaussian", "mixture", "scale_mixture"),
    "n_generate": 2000,
    "n_reverse": 64,
    "reverse_steps": 120,
    "chunk": 64,
    "seed": 0,
}


def _archive(settings):
    from pathlib import Path

    p = Path(settings["archive"])
    return p if p.is_absolute() else PACKAGE_ROOT / p


def _splits(settings):
    train = load_cifar10_luminance(
        _archive(settings), "train", settings["n_train"], settings["seed"]
    )
    test = load_cifar10_luminance(
        _archive(settings), "test", settings["n_test"], settings["seed"],
        stats=(train.mean, train.std),
    )
    return train, test


def _factory(family, settings):
    c = settings["n_components"]
    if family == "gaussian":
        return lambda d, rng: GaussianAR1Kernel(rho=0.2, q=0.8)
    if family == "mixture":
        return lambda d, rng: MixtureInnovationKernel.init(c, rho=0.2, var=0.8, rng=rng)
    if family == "scale_mixture":
        return lambda d, rng: ScaleMixtureKernel.init(c, rho=0.2, var=0.8, rng=rng)
    raise ValueError(f"unknown family {family!r}")


_FIT_CACHE: dict = {}


def _fit_all(settings, train):
    """Fit every family once per process.

    Several parts need the same fitted models, and refitting per part would
    quadruple the cost of a full run. The cache is process-local, so parts
    dispatched as separate array tasks are unaffected and still independent --
    which is the property the `--only` design exists to preserve.
    """
    key = repr(sorted(settings.items()))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    out = {}
    for family in settings["families"]:
        t0 = time.perf_counter()
        model, trace = fit_wavelet_tree(
            train.images, levels=settings["levels"],
            t_train=list(settings["t_train"]),
            kernel_factory=_factory(family, settings),
            n_iters=settings["n_iters"], half_width=settings["half_width"],
            t_resolve=settings["t_resolve"], chunk=settings["chunk"],
        )
        out[family] = (model, trace, time.perf_counter() - t0)
        print(f"[fit] {family}: {out[family][2]:.0f}s, "
              f"{len(trace.log_evidence)} iters, "
              f"monotone violation {trace.monotone_violation:.3g}")
    _FIT_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------

def part_fit(settings, out_dir):
    train, test = _splits(settings)
    fitted = _fit_all(settings, train)
    rows, traces = [], []

    for family, (model, trace, secs) in fitted.items():
        # Evaluate the likelihood only where the grid resolves the likelihood
        # width; below that the coarse subbands are integrated against a mesh
        # that cannot see them. See WaveletTreeModel.resolution_report.
        res = model.resolution_report(settings["t_likelihood"])
        t_ll = max(settings["t_likelihood"], res["min_resolved_t"])
        ll = model.log_likelihood_images(test.images, t_ll, chunk=settings["chunk"])
        row = {
            "family": family,
            "n_train": settings["n_train"],
            "seconds": secs,
            "em_iters": len(trace.log_evidence),
            "monotone_violation": trace.monotone_violation,
            "final_train_log_evidence": trace.log_evidence[-1],
            "t_likelihood_requested": settings["t_likelihood"],
            "t_likelihood_used": t_ll,
            "min_resolved_t": res["min_resolved_t"],
            "min_points_per_std_at_requested_t": res["min_points_per_std"],
            "heldout_loglik_per_image": ll / len(test.images),
        }
        # Every family contributes the same columns; only the scale mixture has a
        # parent-dependent conditional variance, so for the others the fitted
        # magnitude ratio is not merely unknown, it is structurally absent.
        for d in range(model.depth):
            k = model.kernels[0][d]
            row[f"magnitude_ratio_d{d}"] = (
                k.magnitude_ratio(model.grids[d])
                if hasattr(k, "magnitude_ratio") else float("nan")
            )
        rows.append(row)
        traces.extend(
            {"family": family, "iteration": i, "log_evidence": v}
            for i, v in enumerate(trace.log_evidence)
        )
        print(f"[fit] {family}: held-out loglik/image {ll / len(test.images):.3f} "
              f"at t={t_ll:.3f} (min resolved {res['min_resolved_t']:.3f})")

    write_csv(out_dir / "fit.csv", rows)
    write_csv(out_dir / "em_trace.csv", traces)
    return rows


def part_generate(settings, out_dir):
    train, test = _splits(settings)
    fitted = _fit_all(settings, train)
    levels = settings["levels"]

    real_sub = subband_profile(test.images, levels)
    real_cross = crossscale_profile(test.images, levels)
    prof_rows = [{"source": "real_heldout", **r} for r in real_sub]
    cross_rows = [{"source": "real_heldout", **r} for r in real_cross]
    summary = []

    for family, (model, _, _) in fitted.items():
        rng = rng_for("exp25", "generate", family, settings["seed"])
        gen = model.sample_ancestral(settings["n_generate"], rng)
        sub = subband_profile(gen, levels)
        cross = crossscale_profile(gen, levels)
        prof_rows.extend({"source": family, **r} for r in sub)
        cross_rows.extend({"source": family, **r} for r in cross)

        k_gap = profile_gap(sub, real_sub, "excess_kurtosis")
        x_gap = profile_gap(cross, real_cross, "std_ratio_excess_over_null")
        summary.append({
            "family": family,
            "n_generate": settings["n_generate"],
            "kurtosis_mean_abs_gap": k_gap["mean_abs_gap"],
            "kurtosis_mean_signed_gap": k_gap["mean_signed_gap"],
            "kurtosis_worst_abs_gap": k_gap["worst_abs_gap"],
            "magnitude_excess_mean_abs_gap": x_gap["mean_abs_gap"],
            "magnitude_excess_mean_signed_gap": x_gap["mean_signed_gap"],
            "mean_magnitude_excess_generated": float(np.mean(
                [r["std_ratio_excess_over_null"] for r in cross]
            )),
            "mean_magnitude_excess_real": float(np.mean(
                [r["std_ratio_excess_over_null"] for r in real_cross]
            )),
        })
        print(f"[generate] {family}: kurtosis gap {k_gap['mean_abs_gap']:.3f}, "
              f"magnitude-excess gap {x_gap['mean_abs_gap']:.3f} "
              f"(generated {summary[-1]['mean_magnitude_excess_generated']:.3f} "
              f"vs real {summary[-1]['mean_magnitude_excess_real']:.3f})")

    write_csv(out_dir / "subband_profiles.csv", prof_rows)
    write_csv(out_dir / "crossscale_profiles.csv", cross_rows)
    write_csv(out_dir / "generation_summary.csv", summary)
    return summary


def part_sampler(settings, out_dir):
    """Reverse diffusion against ancestral sampling, same model.

    They target the same distribution, so a gap here is discretisation of the
    reverse SDE, not a property of the model. Reported per subband because the
    reverse process resolves coarse structure first and fine structure last, so
    if the sampler is under-resolved the finest subbands are where it shows.
    """
    train, _ = _splits(settings)
    fitted = _fit_all(settings, train)
    rows = []

    for family, (model, _, _) in fitted.items():
        rng = rng_for("exp25", "sampler", family, settings["seed"])
        anc = model.sample_ancestral(settings["n_reverse"], rng)
        t0 = time.perf_counter()
        rev = model.sample_reverse(
            settings["n_reverse"], rng, n_steps=settings["reverse_steps"],
            chunk=settings["chunk"],
        )
        secs = time.perf_counter() - t0
        a_prof = subband_profile(anc, settings["levels"])
        r_prof = subband_profile(rev, settings["levels"])
        for a, r in zip(a_prof, r_prof):
            rows.append({
                "family": family,
                "orientation": a["orientation"],
                "tree_depth": a["tree_depth"],
                "std_ancestral": a["std"],
                "std_reverse": r["std"],
                "kurtosis_ancestral": a["excess_kurtosis"],
                "kurtosis_reverse": r["excess_kurtosis"],
                "std_ratio_reverse_over_ancestral": (
                    r["std"] / a["std"] if a["std"] > 0 else float("nan")
                ),
                "n_samples": settings["n_reverse"],
                "reverse_steps": settings["reverse_steps"],
                "reverse_seconds": secs,
            })
        gap = profile_gap(r_prof, a_prof, "std")
        # Relative, because subband scales span two orders of magnitude and an
        # absolute summary is a report about the LL band and nothing else.
        print(f"[sampler] {family}: reverse vs ancestral std gap "
              f"{gap['mean_rel_gap']:.1%} mean, {gap['worst_rel_gap']:.1%} worst "
              f"(abs {gap['mean_abs_gap']:.4f}), "
              f"{secs:.0f}s for {settings['n_reverse']} samples")

    write_csv(out_dir / "sampler_check.csv", rows)
    return rows


def part_grid(settings, out_dir):
    """Contact sheets. Qualitative, and labelled as such."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test = _splits(settings)
    fitted = _fit_all(settings, train)
    n_show = 8

    panels = [("real (held out)", test.images[:n_show])]
    for family, (model, _, _) in fitted.items():
        rng = rng_for("exp25", "grid", family, settings["seed"])
        panels.append((f"{family} (ancestral)", model.sample_ancestral(n_show, rng)))

    fig, axes = plt.subplots(
        len(panels), n_show, figsize=(n_show * 1.2, len(panels) * 1.35)
    )
    for r, (label, imgs) in enumerate(panels):
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=-2.5, vmax=2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=6, rotation=0,
                              ha="right", va="center")
    fig.suptitle(
        "CIFAR-10 luminance: held-out real vs wavelet-HMT samples "
        f"(n_train={settings['n_train']}, greyscale, 32x32)", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "samples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_dir / 'samples.png'}")
    return []


PARTS = {
    "fit": part_fit,
    "generate": part_generate,
    "sampler": part_sampler,
    "grid": part_grid,
}


def main() -> None:
    parser = experiment_parser(
        "exp_25_wavelet_generation",
        "Generate from the wavelet HMT and compare kernel families on CIFAR-10.",
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings.update(
            n_train=200, n_test=200, n_iters=4, t_resolve=0.2, t_train=(0.5,),
            n_generate=400, n_reverse=8, reverse_steps=30,
        )

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts), "provenance": provenance()})


if __name__ == "__main__":
    main()
