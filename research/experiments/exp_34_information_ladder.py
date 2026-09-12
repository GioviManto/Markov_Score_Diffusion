"""Experiment 34 -- the information ladder (review §10.4, Phase 3).

WHY THIS IS A SMALL ADDITION, NOT A NEW EXPERIMENT
------------------------------------------------------
The review's Phase 3 asks for a ladder of arms, each given one more piece of
structural information than the last, to decompose how much locality,
topology, homogeneity, linear autoregression and the innovation family
separately contribute:

    global MLP                    none
    shared-window network         locality, weight sharing
    bidirectional message passing chain topology, shared local computation
    MDN kernel inside BP          Markov factorisation, homogeneity
    mixture-innovation EM-BP      + linear AR form + low-dim innovation family
    true-kernel BP                complete oracle

Five of these six rungs already exist and are already run, on this exact
protocol, elsewhere in this project: the global MLP and the true-kernel
reference are exp_07's arms, the shared-window and bidirectional arms are
exp_31's, and mixture-innovation EM-BP is exp_31's headline arm and exp_07's.
The one rung nothing in this project fits and scores is `MDNKernel`
(`src/kernels.py`) -- a small MLP whose heads parameterise a per-parent-state
Gaussian mixture, fit inside the exact BP recursion by the same
Fisher's-identity generalised M-step `MixtureInnovationKernel` uses (see
`experiments/exp_14_memorization_collapse.py` for the existing fit pattern).
It is given the Markov factorisation and homogeneity across sites -- because
it is one kernel shared by every edge -- but NOT the linear-autoregressive
form: its mean and scale are arbitrary functions of the parent state, not
`rho * a`.

This experiment fits that one rung, on the same protocol, same bundles, and
(for a shared subset of) the same seeds as exp_31's confirmatory run, so the
two can be read as one table once both land. It is deliberately smaller than
exp_31: fewer seeds, and it reuses the SAME EM-BP fit exp_31 already
certifies rather than refitting it, to keep this an addition to the ladder
rather than a duplicate of the headline comparison.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, resolved_config_hash, select_parts
from frozen_config import FROZEN
from src.bp_grid import grid_bp_batch, make_grid
from src.em import fit_em
from src.kernels import MDNKernel, MixtureInnovationKernel
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = FROZEN.n_sites
RHO = FROZEN.rho
GRID_M = FROZEN.n_grid
GRID_A = FROZEN.half_width
T_SCHEDULE = FROZEN.t_grid

SETTINGS = dict(
    rho=RHO, grid_size=GRID_M, half_width=GRID_A,
    sizes=(32, 128, 512, 2048),
    seeds=8, seed0=0,
    n_val=1024, n_test=2048,
    em_components=8,
    em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200),
    mdn_components=4, mdn_hidden=32,
    mdn_checkpoints=(5, 10, 20, 40, 60, 80, 120, 160, 220, 300, 400),
)

QUICK = dict(
    sizes=(32,), seeds=2, n_val=64, n_test=64, grid_size=201,
    em_checkpoints=(5, 10, 20), mdn_checkpoints=(3, 6, 10),
)


def _bundle(prior, grid, weights, tag, seed, n_chains, t_values):
    rng = rng_for(tag, seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    log_k = prior.log_transition_matrix(grid)
    out = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        out[t] = (X, m_ref)
    return A, out


def _sq(m_hat, m_ref, X, t):
    alpha, delta = alpha_delta(t)
    s_hat = -(X - alpha * m_hat) / delta
    s_ref = -(X - alpha * m_ref) / delta
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())


def _risk(pairs) -> float:
    num, den = sum(e for e, _ in pairs), sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _fit_and_select(cfg, kernel0, groups, grid, weights, val, test, ckpts, tag):
    from src.em import e_step_multi

    _, trace, saved = fit_em(
        kernel0, grid, weights, groups,
        n_iters=max(ckpts), tol=FROZEN.em_loglik_tol, checkpoints=set(ckpts),
    )

    def _eval(kernel, bundle):
        log_k = kernel.log_transition_matrix(grid)
        pairs = []
        for t in T_SCHEDULE:
            X, m_ref = bundle[t]
            al, de = alpha_delta(t)
            m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
            pairs.append(_sq(m, m_ref, X, t))
        return pairs

    best, best_risk = None, np.inf
    for it in sorted(saved):
        r = _risk(_eval(saved[it], val))
        if r < best_risk:
            best, best_risk = it, r
    test_pairs = _eval(saved[best], test)
    return {
        "checkpoint": best, "at_cap": int(best == max(saved)),
        "n_params": int(len(saved[best].theta)),
        "val_risk": best_risk, "test_risk": _risk(test_pairs),
        "converged": trace.converged, "stop_reason": trace.stop_reason,
    }


def part_run(cfg, out):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])

    dest = out / "information_ladder.csv"
    rows, done = [], set()
    if dest.exists():
        import csv as _csv
        rows = list(_csv.DictReader(dest.open()))
        done = {(int(r["seed"]), int(r["n_chains"]), r["arm"]) for r in rows}
        print(f"resuming: {len(done)} cell(s) on disk", flush=True)

    for seed in range(cfg["seed0"], cfg["seed0"] + cfg["seeds"]):
        _, val = _bundle(prior, grid, weights, "exp34-val", seed, cfg["n_val"], T_SCHEDULE)
        _, test = _bundle(prior, grid, weights, "exp34-test", seed, cfg["n_test"], T_SCHEDULE)
        for n_chains in cfg["sizes"]:
            arms_needed = [a for a in ("em_bp", "mdn") if (seed, n_chains, a) not in done]
            if not arms_needed:
                continue
            A = np.stack([
                prior.sample(rng_for("exp34-train", seed, n_chains), N_SITES)
                for _ in range(n_chains)
            ])
            rng_split = rng_for("exp34-split", seed, n_chains)
            idx = np.array_split(rng_split.permutation(n_chains), len(T_SCHEDULE))
            noise = rng_for("exp34-noise", seed, n_chains)
            groups = []
            for t, ix in zip(T_SCHEDULE, idx):
                al, de = alpha_delta(t)
                sub = A[ix]
                groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))

            if "em_bp" in arms_needed:
                k0 = MixtureInnovationKernel.init(
                    cfg["em_components"], rho=0.3, var=0.8,
                    rng=rng_for("exp34-em-init", seed, n_chains),
                )
                res = _fit_and_select(cfg, k0, groups, grid, weights, val, test,
                                       cfg["em_checkpoints"], "em_bp")
                rows.append({"seed": seed, "n_chains": n_chains, "arm": "em_bp", **res})
                print(f"  seed={seed} n={n_chains} em_bp   test_risk={res['test_risk']:.4f} "
                      f"ck={res['checkpoint']}", flush=True)
                write_csv(dest, rows)

            if "mdn" in arms_needed:
                k0 = MDNKernel.init(
                    cfg["mdn_components"], cfg["mdn_hidden"],
                    rng_for("exp34-mdn-init", seed, n_chains),
                )
                res = _fit_and_select(cfg, k0, groups, grid, weights, val, test,
                                       cfg["mdn_checkpoints"], "mdn")
                rows.append({"seed": seed, "n_chains": n_chains, "arm": "mdn", **res})
                print(f"  seed={seed} n={n_chains} mdn     test_risk={res['test_risk']:.4f} "
                      f"ck={res['checkpoint']}", flush=True)
                write_csv(dest, rows)
    write_csv(dest, rows)
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_34_information_ladder",
        "The MDN-inside-BP rung of the information ladder (review Phase 3).",
    )
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {"run": ("fit em_bp and mdn on shared bundles", part_run)}
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {
        "n_sites": N_SITES, "t_schedule": list(T_SCHEDULE),
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg,
    }
    write_json(out / f"params_{tag}.json", {
        **resolved,
        "resolved_config_hash": resolved_config_hash(resolved),
        **provenance(resolved),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
