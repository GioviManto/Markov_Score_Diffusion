"""Experiment 24 -- fit the wavelet hidden Markov tree to CIFAR-10 and score it.

This is the first time the package's exact-BP machinery is applied to data it did
not generate itself. Everything is measured on a held-out split with the
normalisation and the subband scales estimated on training data only.

What this experiment establishes, and what it deliberately does not:

* It establishes that the pipeline runs end to end on real images -- transform,
  per-subband standardisation, exact per-orientation BP with a per-scale noise
  level, per-level M-step -- with exactly monotone EM and a finite exact held-out
  likelihood in *pixel* coordinates.
* It establishes the **Gaussian tree baseline**: the second-order closure that
  every non-Gaussian result has to beat. This is baseline (i) of the plan, and it
  is worth having on its own.
* It does **not** yet test the project's central claim on images. That needs a
  kernel whose conditional *scale* depends on the parent, because
  `exp_23 --only crossscale` shows the cross-scale dependence in CIFAR is largely
  in the magnitude, which a linear-autoregressive kernel cannot represent. Until
  that kernel exists, `mixture` below is reported as what it is: a linear-AR
  model with a flexible innovation *shape*, which improves the marginals and not
  the hierarchy.

Parts
-----
fit       Fit per-scale kernels, record the EM trace and the fitted parameters.
denoise   Held-out denoising MSE against the raw observation and the Gaussian
          closure, across noise levels the model was not fitted at.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from common import (
    PACKAGE_ROOT, apply_overrides, experiment_parser, provenance, select_parts,
)
from src.image_data import load_cifar10_luminance
from src.kernels import GaussianAR1Kernel, MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel, magnitude_diagnostics
from src.noising import alpha_delta
from src.utils import ensure_dir, rng_for, write_csv, write_json
from src.wavelet import ORIENTATIONS
from src.wavelet_model import fit_wavelet_tree

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    "levels": 5,
    "n_train": 2000,
    "n_test": 500,
    "t_train": (0.3, 0.6, 1.0),
    # The denoising sweep. Order carries no meaning -- see `t_loglik`.
    # Extended below 0.2 on 19 Aug to test a prediction: the scale-mixture
    # advantage over the Gaussian closure must vanish at both ends (at t -> 0
    # every model reproduces the observation, at t -> inf every model returns the
    # prior mean), so it peaks somewhere. The calibration showed it falling
    # monotonically across 0.2..1.5, which places the peak at or below 0.2. If
    # the mechanism is the one in exp_23 -- magnitude dependence concentrated at
    # the finest scales, where the per-depth Delta_d = Delta / s_d^2 is largest
    # and the kernel therefore does the most work -- the gain should rise as t
    # falls towards t_resolve. If instead it keeps falling, that story is wrong.
    "t_eval": (0.2, 0.4, 0.8, 1.5, 0.1, 0.06),
    # The single noise level the held-out likelihood is reported at.
    #
    # This used to be `t_eval[1]`, which tied the headline number to an element's
    # *position* in a tuple kept for a different purpose: extending the sweep
    # silently moved the likelihood to another noise level and broke
    # comparability with every earlier run. --quick already diverged this way,
    # reporting at 0.8 against the full run's 0.4. 0.4 is the value every
    # committed result used, so this changes nothing and prevents the next
    # extension from changing something.
    "t_loglik": 0.4,
    "n_iters": 25,
    "t_resolve": 0.05,
    "half_width": 8.0,
    # Raise to refine the FINE-depth meshes only; see fit_wavelet_tree.
    "state_points_per_std": 4.0,
    "n_components": 4,
    # One kernel per level shared across HL/LH/HH, or three.
    #
    # Tying is the default and every result before 20 Aug used it. But exp_23
    # measures linear correlations of 0.452 (HL), 0.482 (LH) and 0.148 (HH) at
    # the finest boundary -- HH is a qualitatively different object, and a tied
    # fit must return one rho for all three. That is a candidate explanation for
    # why the fitted magnitude excess (1.12 at the finest edge) recovers so
    # little of the empirical 1.86-2.33: the structure is being averaged across
    # orientations that do not share it. Untying is the test.
    "tie_orientations": True,
    "families": ("gaussian", "mixture"),
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


def _factory(family: str, settings):
    if family == "gaussian":
        return lambda d, rng: GaussianAR1Kernel(rho=0.2, q=0.8)
    if family == "mixture":
        return lambda d, rng: MixtureInnovationKernel.init(
            settings["n_components"], rho=0.2, var=0.8, rng=rng
        )
    if family == "scale_mixture":
        return lambda d, rng: ScaleMixtureKernel.init(
            settings["n_components"], rho=0.2, var=0.8, rng=rng
        )
    raise ValueError(f"unknown family {family!r}")


def _fit(family, settings, train):
    t0 = time.perf_counter()
    model, trace = fit_wavelet_tree(
        train.images, levels=settings["levels"], t_train=list(settings["t_train"]),
        kernel_factory=_factory(family, settings), n_iters=settings["n_iters"],
        half_width=settings["half_width"], t_resolve=settings["t_resolve"],
        chunk=settings["chunk"], tie_orientations=settings["tie_orientations"],
        state_points_per_std=settings["state_points_per_std"],
    )
    return model, trace, time.perf_counter() - t0


def _aligned(rows):
    """Give every row the same keys, in first-seen order.

    `write_csv` takes its header from the first row alone, and `csv.DictWriter`
    raises on any later row carrying a key that header does not list. The rho
    columns differ by family by construction -- a scalar-rho family has no
    per-component ones and vice versa -- and `families` normally starts with
    `gaussian`, so the scale_mixture row appended after it is exactly the one
    that would raise. Missing entries are left empty rather than filled with a
    number, because "this family has no such parameter" is not a value.
    """
    headers = list(dict.fromkeys(key for row in rows for key in row))
    return [{h: row.get(h, "") for h in headers} for row in rows]


def _kernel_state(k) -> dict:
    """Every fitted parameter of one kernel, as JSON-able numbers.

    Written because `fit.csv` records `rho` and nothing else, so the gate
    parameters that carry the magnitude dependence -- `beta` and `gamma` on
    ScaleMixtureKernel -- died with the process. Any diagnostic not thought of
    in advance then costs a refit, which for this experiment is hours. The
    kernels are frozen dataclasses, so the field list is the parameter list.
    """
    if not dataclasses.is_dataclass(k):
        return {"repr": repr(k)}
    out = {"class": type(k).__name__}
    for f in dataclasses.fields(k):
        value = getattr(k, f.name)
        arr = np.asarray(value, dtype=float)
        out[f.name] = arr.tolist() if arr.ndim else float(arr)
    return out


def part_fit(settings, out_dir):
    train, test = _splits(settings)
    rows, traces = [], []
    kernel_params: dict = {}

    per_image: dict = {}
    for family in settings["families"]:
        model, trace, secs = _fit(family, settings, train)
        # Per-image, not just the total. Two families scored on the SAME held-out
        # images give a paired difference whose standard error is far below either
        # family's spread across images, because image difficulty is common to
        # both and cancels. Without this vector a likelihood gap has no error bar,
        # and "family A beats family B by 14.8 nats" cannot be told apart from
        # noise. It is also the comparison FID cannot make: FID's own real-vs-real
        # floor is 4.5 at n=10^4 (exp_30), so a percent-level model difference is
        # under its resolution while being many sigma in likelihood.
        ll_each = model.log_likelihood_images(
            test.images, settings["t_loglik"], per_image=True
        )
        per_image[family] = np.asarray(ll_each, dtype=float)
        ll = float(np.sum(ll_each))
        row = {
            "family": family,
            "n_train": settings["n_train"],
            "seconds": secs,
            "em_iters": len(trace.log_evidence),
            "final_log_evidence": trace.log_evidence[-1],
            "monotone_violation": trace.monotone_violation,
            "heldout_loglik_per_image": ll / len(test.images),
        }
        for d in range(model.depth):
            k = model.kernels[0][d]
            # `d` is already the detail level: `kernels[orientation][d]` holds one
            # kernel per (orientation, level). What varies *inside* one kernel is
            # the mixture component. GaussianAR1Kernel and MixtureInnovationKernel
            # carry a scalar `rho` shared by every component; ScaleMixtureKernel
            # carries one `rho_c` per component, shape (C,), and float() on that
            # array is what killed job 627175 on 2026-08-14. Give a vector kernel
            # one column per component instead of collapsing it -- differing
            # components are the whole point of the family, and a mean over them
            # would report a number the model never uses.
            rho = np.atleast_1d(getattr(k, "rho", np.nan)).astype(float)
            if rho.size == 1:
                row[f"rho_d{d}"] = float(rho[0])
            else:
                for c, value in enumerate(rho):
                    row[f"rho_d{d}_c{c}"] = float(value)

            # The measurement this whole experiment is for. `exp_23` found an
            # empirical Q4/Q1 magnitude excess of 1.86-2.33 at the finest scale
            # boundary, and a scale-mixture kernel exists because a linear-AR one
            # cannot represent any of it. Whether the *fitted* kernel reproduces
            # that excess is the question -- and until now it was neither computed
            # here nor recoverable afterwards, because only `rho` was persisted
            # and the gate parameters that carry the dependence were discarded
            # with the model. A held-out likelihood cannot answer it: a family
            # with more parameters can win on likelihood while leaving the
            # magnitude structure untouched.
            #
            # Wrapped, because a diagnostic must never be able to destroy the run
            # it is diagnosing. This is the last statement before an 8-hour fit's
            # results are written; a nan in a column is recoverable, a raise here
            # loses every family in the job.
            try:
                row.update({
                    f"{key}_d{d}": value
                    for key, value in magnitude_diagnostics(k, model.grids[d]).items()
                })
                # Untied fits carry a different kernel per orientation, and the
                # orientations are the whole point: exp_23 finds HH with linear
                # correlation 0.148 against HL's 0.452 but a comparable magnitude
                # excess, so an HH kernel that has learned the structure should
                # show a *low* implied rho and a *high* excess. Averaging the
                # three, or reporting only the first, would hide precisely that.
                if not settings["tie_orientations"]:
                    for oi, name in enumerate(ORIENTATIONS):
                        diag = magnitude_diagnostics(
                            model.kernels[oi][d], model.grids[d]
                        )
                        row.update({
                            f"{key}_{name}_d{d}": value
                            for key, value in diag.items()
                        })
            except Exception as exc:                       # pragma: no cover
                row[f"magnitude_error_d{d}"] = repr(exc)
        kernel_params[family] = {
            f"{name}_depth_{d}": _kernel_state(model.kernels[oi][d])
            for d in range(model.depth)
            for oi, name in enumerate(ORIENTATIONS)
        }
        kernel_params[family]["grid_sizes"] = [int(g.size) for g in model.grids]
        rows.append(row)
        traces.extend(
            {"family": family, "iteration": i, "log_evidence": v}
            for i, v in enumerate(trace.log_evidence)
        )
        print(f"[fit] {family}: {secs:.0f}s, {len(trace.log_evidence)} iters, "
              f"monotone violation {trace.monotone_violation:.3g}, "
              f"held-out loglik/image {ll / len(test.images):.2f}")

    write_csv(out_dir / "fit.csv", _aligned(rows))
    write_csv(out_dir / "em_trace.csv", traces)
    write_json(out_dir / "kernels.json", kernel_params)

    if per_image:
        fams = list(per_image)
        np.savez(out_dir / "loglik_per_image.npz", **per_image)
        # The paired comparison, computed here so it cannot be done wrongly
        # later: pair on image, never compare two unpaired means.
        pair_rows = []
        for i, a in enumerate(fams):
            for b in fams[i + 1:]:
                d = per_image[a] - per_image[b]
                n = len(d)
                se = float(np.std(d, ddof=1) / np.sqrt(n))
                pair_rows.append({
                    "family_a": a, "family_b": b, "n_test": n,
                    "mean_diff_nats_per_image": float(np.mean(d)),
                    "paired_se": se,
                    "t": float(np.mean(d) / se) if se > 0 else float("nan"),
                    "a_better_on_images": int(np.sum(d > 0)),
                    # What an UNPAIRED comparison would have reported, to show
                    # what the pairing buys.
                    "unpaired_se": float(np.sqrt(
                        np.var(per_image[a], ddof=1) + np.var(per_image[b], ddof=1)
                    ) / np.sqrt(n)),
                })
                print(f"[loglik] {a} - {b}: "
                      f"{np.mean(d):+.2f} +/- {se:.2f} nats/image "
                      f"(t={pair_rows[-1]['t']:.1f}, "
                      f"{pair_rows[-1]['a_better_on_images']}/{n} images)")
        # A single-family run has no pairs, and `write_csv` refuses an empty
        # list. Guarding rather than letting it raise: the pairing is a
        # convenience, and it crashed jobs 633540-2 AFTER a 15-minute fit had
        # already succeeded and been written. A diagnostic must not be able to
        # fail a run whose results are already on disk.
        if pair_rows:
            write_csv(out_dir / "loglik_paired.csv", pair_rows)
    return rows


def part_denoise(settings, out_dir):
    """Held-out denoising, including at noise levels not used for fitting.

    The point of the construction is that one fitted kernel is a denoiser at
    *every* t with no refitting, so t_eval deliberately straddles t_train.
    """
    train, test = _splits(settings)
    rng = rng_for("exp24", "denoise", settings["seed"])
    rows = []

    fitted = {f: _fit(f, settings, train)[0] for f in settings["families"]}

    for t in settings["t_eval"]:
        alpha, delta = alpha_delta(t)
        noisy = alpha * test.images + np.sqrt(delta) * rng.standard_normal(
            test.images.shape
        )
        row = {
            "t": t,
            "alpha": alpha,
            "delta": delta,
            "fitted_at_this_t": t in settings["t_train"],
            "mse_raw_rescaled": float(np.mean((noisy / alpha - test.images) ** 2)),
        }
        for family, model in fitted.items():
            hat = model.denoise_images(noisy, t, chunk=settings["chunk"])
            row[f"mse_{family}"] = float(np.mean((hat - test.images) ** 2))
            row[f"loglik_per_image_{family}"] = (
                model.log_likelihood_images(noisy, t, chunk=settings["chunk"])
                / len(test.images)
            )
        rows.append(row)
        print(f"[denoise] t={t}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k.startswith("mse_")
        ))

    write_csv(out_dir / "denoise.csv", rows)
    return rows


PARTS = {"fit": part_fit, "denoise": part_denoise}


def main() -> None:
    parser = experiment_parser(
        "exp_24_wavelet_fit",
        "Fit the wavelet HMT to CIFAR-10 and score it on held-out images.",
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings.update(
            n_train=300, n_test=100, n_iters=6, t_resolve=0.3,
            t_train=(0.5,), t_eval=(0.4, 0.8), t_loglik=0.4,
        )

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts), "provenance": provenance()})


if __name__ == "__main__":
    main()
