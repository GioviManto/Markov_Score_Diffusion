"""Experiment 26 -- video: what exact BP on a caterpillar can and cannot do.

The model is a temporal chain of spatial quadtrees (`src/video_bp.py`): frames
coupled through the *root* of each frame's tree, which is the loop-free spanning
structure of the spatio-temporal graph. BP on it is exact. The question this
experiment answers is whether that exactness is worth what it costs.

Two models, identical in every respect but one:

    caterpillar   temporal kernel fitted by EM through the caterpillar E-step.
    frozen        temporal kernel held at rho_time = 0 -- frames independent
                  *given* the LL trajectory, which is shared by both.

Isolating the comparison to the tree-root coupling is the point. The LL band is
handled identically in both (an exact scalar Gaussian AR(1) across frames), so
the difference between them is the one thing being tested and not a bundle.

The metric that matters is **frame-difference energy**: mean squared successive
frame difference over frame variance. 0 is a frozen sequence, 2 is successive
frames independent. Real translating video sits near 0.22.

Parts
-----
fit        Both models; held-out sequence likelihood, fitted rho_time.
coherence  Frame-difference energy of generated video against real, plus the
           variance-decomposition prediction that says what the structure alone
           implies.
strip      PNG of real and generated frame strips.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    PACKAGE_ROOT, apply_overrides, experiment_parser, provenance, select_parts,
)
from src.image_data import load_cifar10_luminance
from src.kernels import GaussianAR1Kernel
from src.scale_kernel import ScaleMixtureKernel
from src.utils import ensure_dir, rng_for, write_csv, write_json
from src.video_data import frame_difference_energy, make_moving_sequences
from src.video_model import fit_video_tree
from src.wavelet import images_to_tree

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    "levels": 5,
    "n_train": 300,
    "n_test": 200,
    "n_frames": 6,
    "max_speed": 1,
    "t_train": (0.6,),
    "t_likelihood": 0.6,
    "n_iters": 8,
    "t_resolve": 0.05,
    "half_width": 8.0,
    "n_components": 4,
    "spatial_family": "gaussian",
    "n_generate": 400,
    "cuts": (0, 1, 2, 3, 4),
    "chunk": 16,
    "seed": 0,
}


def _archive(settings):
    from pathlib import Path

    p = Path(settings["archive"])
    return p if p.is_absolute() else PACKAGE_ROOT / p


def _data(settings):
    train = load_cifar10_luminance(
        _archive(settings), "train", settings["n_train"], settings["seed"]
    )
    test = load_cifar10_luminance(
        _archive(settings), "test", settings["n_test"], settings["seed"],
        stats=(train.mean, train.std),
    )
    rng = rng_for("exp26", "motion", settings["seed"])
    tr = make_moving_sequences(
        train.images, settings["n_frames"], rng, settings["max_speed"]
    )
    te = make_moving_sequences(
        test.images, settings["n_frames"], rng, settings["max_speed"]
    )
    return tr, te


def _spatial_factory(settings):
    if settings["spatial_family"] == "scale_mixture":
        return lambda d, rng: ScaleMixtureKernel.init(
            settings["n_components"], rho=0.2, var=0.8, rng=rng
        )
    return lambda d, rng: GaussianAR1Kernel(rho=0.2, q=0.8)


_CACHE: dict = {}


def _fit_both(settings, train):
    key = repr(sorted(settings.items()))
    if key in _CACHE:
        return _CACHE[key]
    out = {}
    for name, freeze, rho0 in (("caterpillar", False, 0.3), ("frozen", True, 0.0)):
        t0 = time.perf_counter()
        model, trace = fit_video_tree(
            train, levels=settings["levels"], t_train=list(settings["t_train"]),
            kernel_factory=_spatial_factory(settings),
            time_kernel_factory=lambda r, _r=rho0: GaussianAR1Kernel(rho=_r, q=0.8),
            n_iters=settings["n_iters"], half_width=settings["half_width"],
            t_resolve=settings["t_resolve"], freeze_time=freeze,
            chunk=settings["chunk"],
        )
        out[name] = (model, trace, time.perf_counter() - t0)
        print(f"[fit] {name}: {out[name][2]:.0f}s, rho_time "
              f"{model.k_time.rho:.4f}, monotone violation "
              f"{trace.monotone_violation:.3g}")
    _CACHE[key] = out
    return out


def _coupled_variance_fraction(videos, levels):
    """Fraction of per-frame variance carried by the temporally coupled parts.

    The caterpillar couples the *roots* of the three orientation trees and the LL
    band. Everything else is redrawn independently at every frame, so this
    fraction is what caps the achievable temporal coherence before any parameter
    is fitted.
    """
    flat = videos.reshape(-1, *videos.shape[2:])
    qt, nodes, scaling = images_to_tree(flat, levels)
    depth_of = qt.node_depth
    total = 0.0
    root = 0.0
    for oi in range(3):
        for d in range(qt.depth + 1):
            blk = nodes[:, oi, depth_of == d]
            v = float(blk.var() * blk.shape[1])
            total += v
            if d == 0:
                root += v
    ll = float(scaling.var() * scaling.shape[1])
    return (root + ll) / (total + ll), root / (total + ll), ll / (total + ll)


def part_fit(settings, out_dir):
    train, test = _data(settings)
    fitted = _fit_both(settings, train)
    rows, traces = [], []

    for name, (model, trace, secs) in fitted.items():
        # Only quote the likelihood where the grid resolves the likelihood width.
        # Below that the coarse subbands are integrated against a mesh that
        # cannot see them, exactly as in exp_25.
        res = model.resolution_report(settings["t_likelihood"])
        t_ll = max(settings["t_likelihood"], res["min_resolved_t"])
        ll = model.log_likelihood_videos(test, t_ll, chunk=settings["chunk"])
        rows.append({
            "model": name,
            "n_train": settings["n_train"],
            "n_frames": settings["n_frames"],
            "seconds": secs,
            "rho_time": float(model.k_time.rho),
            "rho_ll": model.ll_rho,
            "monotone_violation": trace.monotone_violation,
            "t_likelihood_requested": settings["t_likelihood"],
            "t_likelihood_used": t_ll,
            "min_resolved_t": res["min_resolved_t"],
            "heldout_loglik_per_sequence": ll / len(test),
            "heldout_loglik_per_frame": ll / (len(test) * settings["n_frames"]),
        })
        traces.extend(
            {"model": name, "iteration": i, "log_evidence": v}
            for i, v in enumerate(trace.log_evidence)
        )
        print(f"[fit] {name}: held-out loglik/sequence {ll / len(test):.2f} "
              f"at t={t_ll:.3f} (min resolved {res['min_resolved_t']:.3f})")

    write_csv(out_dir / "fit.csv", rows)
    write_csv(out_dir / "em_trace.csv", traces)
    return rows


def part_coherence(settings, out_dir):
    train, test = _data(settings)
    fitted = _fit_both(settings, train)
    real = frame_difference_energy(test)
    frac, frac_root, frac_ll = _coupled_variance_fraction(train, settings["levels"])
    rows = []

    for name, (model, _, _) in fitted.items():
        rng = rng_for("exp26", "gen", name, settings["seed"])
        gen = model.sample_ancestral(
            settings["n_generate"], settings["n_frames"], rng
        )
        got = frame_difference_energy(gen)
        # What the structure alone implies, before any fitting: the coupled
        # fraction contributes 2(1 - rho) and the rest contributes the full 2.
        pred = (
            2.0 * (1.0 - model.k_time.rho) * frac_root
            + 2.0 * (1.0 - model.ll_rho) * frac_ll
            + 2.0 * (1.0 - frac)
        )
        rows.append({
            "model": name,
            "rho_time": float(model.k_time.rho),
            "frame_diff_energy_real": real,
            "frame_diff_energy_generated": got,
            "frame_diff_energy_structural_prediction": pred,
            "frame_diff_energy_independent_frames": 2.0,
            "coupled_variance_fraction": frac,
            "prediction_error": abs(got - pred),
        })
        print(f"[coherence] {name}: generated {got:.4f}, predicted {pred:.4f}, "
              f"real {real:.4f} (independent frames = 2.0)")

    print(f"[coherence] temporally coupled variance fraction: {frac:.4f}")
    write_csv(out_dir / "coherence.csv", rows)
    return rows


def part_strip(settings, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test = _data(settings)
    fitted = _fit_both(settings, train)
    f = settings["n_frames"]

    panels = [("real", test[:2])]
    for name, (model, _, _) in fitted.items():
        rng = rng_for("exp26", "strip", name, settings["seed"])
        panels.append((name, model.sample_ancestral(2, f, rng)))

    n_rows = len(panels) * 2
    fig, axes = plt.subplots(n_rows, f, figsize=(f * 1.1, n_rows * 1.2))
    r = 0
    for label, vids in panels:
        for k in range(2):
            for c in range(f):
                ax = axes[r, c]
                ax.imshow(vids[k, c], cmap="gray", vmin=-2.5, vmax=2.5)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(label, fontsize=6, rotation=0,
                                  ha="right", va="center")
            r += 1
    fig.suptitle("Moving CIFAR: real vs caterpillar samples (time runs left to right)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "video_strips.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[strip] wrote {out_dir / 'video_strips.png'}")
    return []


def part_structure(settings, out_dir):
    """The ceiling, derived rather than fitted.

    Two things, neither of which involves training a model:

    * the loop-free budget -- at most one temporal edge per per-frame connected
      component, established by union-find over every choice rather than by
      argument;
    * the coherence floor that budget implies, i.e. the frame-difference energy
      of a model whose temporal kernels are *perfect*.

    Because the floor is reached at rho = 1, it bounds every loop-free
    spatio-temporal model over this coefficient set, not merely the fitted one.
    """
    import itertools

    from src.hierarchy import TreeIndex
    from src.video_bp import spatiotemporal_has_cycle

    train, test = _data(settings)

    # -- the budget --------------------------------------------------------
    ti = TreeIndex(settings["levels"] - 1, 4)
    edges = [
        (int(node), int(child))
        for d in range(ti.depth)
        for node in ti.nodes_at(d)
        for child in ti.children(int(node), d)
    ]
    budget = []
    for k in (1, 2, 3):
        cyclic = [
            spatiotemporal_has_cycle(ti.n_nodes, settings["n_frames"], edges, c)
            for c in itertools.islice(itertools.combinations(range(ti.n_nodes), k), 40)
        ]
        budget.append({
            "coupled_nodes_per_component": k,
            "any_choice_loop_free": bool(not all(cyclic)),
            "all_choices_loop_free": bool(not any(cyclic)),
        })
        print(f"[structure] {k} temporal edge(s) per component: "
              f"loop-free for all choices = {budget[-1]['all_choices_loop_free']}")

    # -- the floor ---------------------------------------------------------
    frac, frac_root, frac_ll = _coupled_variance_fraction(train, settings["levels"])
    real = frame_difference_energy(test)
    rows = [{
        "coupled_components": 4,
        "coupled_coefficients_per_frame": 4,
        "coefficients_per_frame": 3 * ti.n_nodes + 1,
        "coupled_variance_fraction": frac,
        "floor_frame_diff_energy": 2.0 * (1.0 - frac),
        "real_frame_diff_energy": real,
        "independent_frames": 2.0,
        "floor_over_real": 2.0 * (1.0 - frac) / real if real > 0 else float("nan"),
    }]
    print(f"[structure] coupled variance {frac:.4f} over 4 of "
          f"{3 * ti.n_nodes + 1} coefficients")
    print(f"[structure] FLOOR with perfect temporal kernels: "
          f"{2.0 * (1.0 - frac):.4f}  vs real {real:.4f} "
          f"({2.0 * (1.0 - frac) / real:.1f}x)")

    write_csv(out_dir / "structure_budget.csv", budget)
    write_csv(out_dir / "structure_floor.csv", rows)
    return rows


def _coupled_fraction_at_cut(videos, levels, cut):
    """Variance fraction a cut-`cut` forest can couple in time.

    Cutting below depth c leaves the top piece (whose best single node is the
    depth-0 root) and one component per depth-c node, so *every* depth-c
    coefficient becomes couplable. Hence root variance plus the whole of level c,
    plus LL, which is its own component either way.
    """
    flat = videos.reshape(-1, *videos.shape[2:])
    qt, nodes, scaling = images_to_tree(flat, levels)
    depth_of = qt.node_depth
    per_depth = np.zeros(qt.depth + 1)
    for oi in range(3):
        for d in range(qt.depth + 1):
            blk = nodes[:, oi, depth_of == d]
            per_depth[d] += float(blk.var() * blk.shape[1])
    ll = float(scaling.var() * scaling.shape[1])
    total = per_depth.sum() + ll
    coupled = per_depth[0] + (0.0 if cut == 0 else per_depth[cut])
    return (coupled + ll) / total


def part_cutcurve(settings, out_dir):
    """The space-for-time trade, measured rather than predicted.

    Cutting the spatial tree buys temporal edges but *removes real spatial
    dependence*, so coherence and likelihood pull in opposite directions. The
    likelihood is the arbiter: a cut that improves coherence while costing
    likelihood is a more coherent model, not a better one. Both are reported and
    neither is allowed to stand for the other.
    """
    train, test = _data(settings)
    depth = settings["levels"] - 1
    rows = []

    # `cuts` is a setting so a job array can take one depth per task: each fit is
    # independent and writes its own file, so there is no merge step.
    for cut in settings["cuts"]:
        if not 0 <= cut <= depth:
            raise SystemExit(f"cut {cut} outside [0, {depth}]")
        t0 = time.perf_counter()
        model, trace = fit_video_tree(
            train, levels=settings["levels"], t_train=list(settings["t_train"]),
            kernel_factory=_spatial_factory(settings),
            time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.3, q=0.8),
            n_iters=settings["n_iters"], half_width=settings["half_width"],
            t_resolve=settings["t_resolve"],
            cut_depth=cut, chunk=settings["chunk"],
        )
        secs = time.perf_counter() - t0
        res = model.resolution_report(settings["t_likelihood"])
        t_ll = max(settings["t_likelihood"], res["min_resolved_t"])
        ll = model.log_likelihood_videos(test, t_ll, chunk=settings["chunk"])
        rng = rng_for("exp26", "cutgen", cut, settings["seed"])
        gen = model.sample_ancestral(
            settings["n_generate"], settings["n_frames"], rng
        )
        fde = frame_difference_energy(gen)
        frac = _coupled_fraction_at_cut(train, settings["levels"], cut)
        rows.append({
            "cut_depth": cut,
            "components_per_orientation": 1 if cut == 0 else 1 + 4**cut,
            "spatial_edges_severed_per_frame": 0 if cut == 0 else 3 * 4**cut,
            "coupled_variance_fraction": frac,
            "coherence_floor": 2.0 * (1.0 - frac),
            "rho_time_top": float(model.k_time.rho),
            "rho_time_sub": (
                float(model.k_time_sub.rho) if model.k_time_sub is not None
                else float("nan")
            ),
            "monotone_violation": trace.monotone_violation,
            "t_likelihood_used": t_ll,
            "heldout_loglik_per_sequence": ll / len(test),
            "frame_diff_energy_generated": fde,
            "frame_diff_energy_real": frame_difference_energy(test),
            "seconds": secs,
        })
        print(f"[cutcurve] cut={cut}: loglik/seq {ll / len(test):9.2f}  "
              f"fde {fde:.4f} (floor {2.0 * (1.0 - frac):.4f})  "
              f"rho_sub {rows[-1]['rho_time_sub']:.4f}  {secs:.0f}s", flush=True)

    best_ll = max(rows, key=lambda r: r["heldout_loglik_per_sequence"])
    best_fde = min(rows, key=lambda r: r["frame_diff_energy_generated"])
    print(f"[cutcurve] best likelihood at cut={best_ll['cut_depth']}, "
          f"best coherence at cut={best_fde['cut_depth']}")
    suffix = "" if tuple(settings["cuts"]) == tuple(range(depth + 1)) else \
        "_cut" + "".join(str(c) for c in settings["cuts"])
    write_csv(out_dir / f"cutcurve{suffix}.csv", rows)
    return rows


PARTS = {
    "fit": part_fit,
    "cutcurve": part_cutcurve,
    "coherence": part_coherence,
    "structure": part_structure,
    "strip": part_strip,
}


def main() -> None:
    parser = experiment_parser(
        "exp_26_video", "Exact BP on a temporal chain of spatial trees, on moving CIFAR."
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings.update(
            n_train=80, n_test=60, n_iters=3, n_generate=100, t_resolve=0.3
        )

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts), "provenance": provenance()})


if __name__ == "__main__":
    main()
