"""Experiment 29 -- does EM overfit, and where should it be stopped?

This exists because a hard-coded iteration budget in exp_07 turned out to matter
in a way nobody had checked. Comparing 120 against 400 iterations on identical
data and identical networks, the estimator got *worse* at small n and better at
large n. Two explanations fit that pattern and they call for opposite fixes:

  (a) overfitting -- EM maximises the marginal likelihood of the TRAINING
      sequences, and past some point the extra fit is to the sample, not the law;
  (b) a bad optimum -- longer runs wander into a worse stationary point.

They are distinguishable, and only by instrumenting both objectives on the SAME
iterates. Under (a) the training log-evidence keeps climbing -- EM guarantees it
must -- while held-out error turns around. Under (b) the training objective would
stall or fall too. Reporting one without the other cannot tell them apart, which
is exactly the failure mode Chapter "What we got wrong" is full of.

The answer is (a), unambiguously: training evidence is monotone to the last
iterate at every size, and held-out error has an interior minimum at every size,
including the largest. A fixed cap therefore overshoots everywhere, which is why
exp_07 now selects the stopping point on a validation bundle.

Deliberately NOT claimed here: how the optimal stopping point scales with n. The
sizes below are chosen to bracket the effect, not to trace a curve, and reading a
scaling law off three points is the mistake this project has already made once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import make_grid
from src.denoiser import bp_posterior_mean, evaluate_denoiser
from src.em import e_step_multi
from src.kernels import MixtureInnovationKernel
from src.priors import LaplaceAR1
from src.utils import ensure_dir, write_csv, write_json
from frozen_config import FROZEN

import exp_07_em_vs_score_network as E7

FROZEN_ROOT = Path("outputs/frozen")


def part_trace(cfg: dict) -> list[dict]:
    """Both objectives, on the same iterates, at each size."""
    grid, weights = make_grid(FROZEN.half_width, FROZEN.n_grid)
    prior = LaplaceAR1(E7.RHO_TRUE)
    _, bundle = E7.make_test_set(
        prior, grid, weights, E7.T_TRAIN, cfg["n_heldout"])

    rows: list[dict] = []
    for n_chains in cfg["sizes"]:
        # Same RNG keys as exp_07 part 1, so this traces the very fits that
        # experiment reports rather than a lookalike.
        rng = E7.train_rng("exp07-p1", n_chains)
        A = np.stack([prior.sample(rng, E7.N_SITES) for _ in range(n_chains)])
        groups = E7.noisy_groups(A, E7.T_TRAIN, rng)

        k = MixtureInnovationKernel.init(
            E7.N_COMPONENTS, rho=0.3, var=0.8, rng=E7.train_rng("exp07-init"))

        for it in range(cfg["n_iters"] + 1):
            stats = e_step_multi(grid, weights, k.log_transition_matrix(grid), groups)
            if it % cfg["every"] == 0:
                errs = [
                    evaluate_denoiser(
                        bp_posterior_mean(k, grid, weights, bundle[t][0], t),
                        bundle[t][1], bundle[t][0], t)["score_rel_l2"]
                    for t in E7.T_TRAIN
                ]
                rows.append({
                    "n_chains": n_chains,
                    "iter": it,
                    # Per sequence, so the two sizes are on one scale.
                    "train_logev_per_seq": stats.log_evidence / n_chains,
                    "heldout_score_rel_l2": float(np.mean(errs)),
                })
                print(f"  n={n_chains:5d} it={it:4d} "
                      f"train={stats.log_evidence / n_chains:12.5f} "
                      f"heldout={np.mean(errs):.5f}", flush=True)
            if it < cfg["n_iters"]:
                k = k.m_step(stats, grid)
    return rows


def part_summary(rows: list[dict]) -> list[dict]:
    """The verdict: is training monotone, and where is held-out best?"""
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["n_chains"]].append(r)

    out = []
    for n, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["iter"])
        tr = np.array([r["train_logev_per_seq"] for r in rs])
        ho = np.array([r["heldout_score_rel_l2"] for r in rs])
        best = int(rs[int(ho.argmin())]["iter"])
        out.append({
            "n_chains": n,
            "train_monotone_increasing": bool(np.all(np.diff(tr) >= -1e-9)),
            "heldout_best_iter": best,
            "heldout_best": float(ho.min()),
            "heldout_at_cap": float(ho[-1]),
            "cost_of_running_to_cap_pct": float(100 * (ho[-1] / ho.min() - 1)),
            # An interior optimum is the whole point: at the boundary the trace
            # would only say the budget was too short to see a turn.
            "optimum_is_interior": bool(0 < int(ho.argmin()) < len(ho) - 1),
        })
    return out


def main() -> None:
    parser = experiment_parser(
        "exp_29_em_overfitting",
        "Training evidence against held-out error on the same EM iterates.",
    )
    args = parser.parse_args()

    settings = {
        # Two below and one well above the point where 120 and 400 swap order,
        # so a size-independent effect can be told from a size-dependent one.
        "sizes": [32, 64, 2048],
        "n_iters": FROZEN.em_max_iters,
        "every": 10,
        "n_heldout": FROZEN.n_heldout,
    }
    if args.quick:
        settings.update(sizes=[32], n_iters=40, every=10, n_heldout=64)
    cfg = apply_overrides(settings, args.set)

    rows_holder: dict = {}

    def _trace(out):
        rows_holder["rows"] = part_trace(cfg)
        write_csv(out / "overfitting_trace.csv", rows_holder["rows"])

    def _summary(out):
        rows = rows_holder.get("rows") or part_trace(cfg)
        summary = part_summary(rows)
        write_csv(out / "overfitting_summary.csv", summary)
        for s in summary:
            print(f"  n={s['n_chains']:5d} train monotone {s['train_monotone_increasing']} "
                  f" best at {s['heldout_best_iter']:4d} "
                  f" running to cap costs {s['cost_of_running_to_cap_pct']:+.1f}%",
                  flush=True)

    parts = {
        "trace": ("both objectives on the same iterates", _trace),
        "summary": ("the verdict: monotone training, interior held-out optimum", _summary),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir
                     if args.output_dir != parser.get_default("output_dir")
                     else FROZEN_ROOT / "exp_29_em_overfitting")
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    write_json(out / "params.json", {**cfg, **provenance()})
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
