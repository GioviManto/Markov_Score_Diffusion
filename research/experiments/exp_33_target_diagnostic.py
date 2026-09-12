"""Experiment 33 -- why does the network's error flatten? (review §10.3, Phase 2)

WHY THIS IS SEPARATE FROM exp_31, AND WHY IT IS SMALLER
---------------------------------------------------------
exp_31 already removes the confound that made this diagnostic necessary for
the paper's CORRECTNESS: both arms there checkpoint and select on validation
rather than training for a fixed step count, so a widening gap can no longer
be silently attributed to a shrinking per-datum budget. The manuscript's
retraction of the "convolution fails to acquire structure" mechanism claim
does not depend on this experiment.

What exp_31 does not do is say WHY a residual gap -- if one survives -- would
exist. The review's §10.3 asks for that mechanism, as a 2x2 separating four
candidate explanations:

    fixed finite training set   x   DSM noise/x0 target
    fresh chains every step     x   exact BP posterior-mean target

  * Fixed data + DSM target:      ordinary training. Confounds all three
                                   effects below into one number.
  * Fresh data + DSM target:      removes finite-DATA estimation error (the
                                   network never reuses a training example).
  * Fixed data + exact target:    removes DSM TARGET variance (the network
                                   regresses onto E[a|x] directly rather than
                                   a single noisy realisation).
  * Fresh data + exact target:    removes both. What remains is architecture
                                   /optimisation error against the population
                                   objective, which is the only cell that
                                   licenses a statement resembling "the
                                   architecture cannot represent this".

This is explanatory, not a correctness gate -- the review's own §12 places it
under "high-value within the remaining time", not blocking -- so it runs at
one architecture (the bidirectional message-passing net: `src/seq_nets.py`
motivates it as the strongest claim to representing what BP computes), one
size (n=2048, where the original fixed-budget confound was largest), and
fewer seeds than exp_31. A demonstrative mechanism check, not a second
headline table.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, resolved_config_hash, select_parts
from frozen_config import FROZEN
from src.bp_grid import grid_bp_batch, make_grid
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.seq_nets import BiMessagePassing, posterior_mean, train_sequence_net
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = FROZEN.n_sites
RHO = FROZEN.rho
GRID_M = FROZEN.n_grid
GRID_A = FROZEN.half_width
T_SCHEDULE = FROZEN.t_grid

SETTINGS = dict(
    rho=RHO, grid_size=GRID_M, half_width=GRID_A,
    n_chains=2048,
    n_val=1024, n_test=2048,
    seeds=6,
    seed0=0,
    hidden=64,
    lr=1e-3,
    parameterization="eps",
    checkpoints=(500, 1000, 2000, 4000, 8000, 16000, 32000),
    batch_size=64,
    grad_clip=1.0,
    conditions=("fixed_dsm", "fresh_dsm", "fixed_exact", "fresh_exact"),
)

QUICK = dict(
    n_chains=128, n_val=64, n_test=64, seeds=2, hidden=16,
    checkpoints=(50, 200), grid_size=201,
)


def _region_pairs(kernel_or_model, grid, weights, bundle, predict_fn, region=slice(0, None)):
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        alpha, delta = alpha_delta(t)
        m_hat = predict_fn(X, t)
        s_hat = -(X[:, region] - alpha * m_hat[:, region]) / delta
        s_ref = -(X[:, region] - alpha * m_ref[:, region]) / delta
        pairs.append((float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())))
    return pairs


def _risk(pairs) -> float:
    num, den = sum(e for e, _ in pairs), sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _bundle(prior, grid, weights, tag, seed, n_chains):
    rng = rng_for(tag, seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    log_k = prior.log_transition_matrix(grid)
    out = {}
    for t in T_SCHEDULE:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        out[t] = (X, m_ref)
    return A, out


def part_run(cfg, out):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    log_k_true = prior.log_transition_matrix(grid)

    dest = out / "target_diagnostic.csv"
    rows, done = [], set()
    if dest.exists():
        import csv as _csv
        rows = list(_csv.DictReader(dest.open()))
        done = {(int(r["seed"]), r["condition"]) for r in rows}
        print(f"resuming: {len(done)} cell(s) on disk", flush=True)

    for seed in range(cfg["seed0"], cfg["seed0"] + cfg["seeds"]):
        A_fixed = np.stack([
            prior.sample(rng_for("exp33-train", seed), N_SITES)
            for _ in range(cfg["n_chains"])
        ])
        _, val = _bundle(prior, grid, weights, "exp33-val", seed, cfg["n_val"])
        _, test = _bundle(prior, grid, weights, "exp33-test", seed, cfg["n_test"])

        def exact_target(X, t):
            alpha, delta = alpha_delta(t)
            m, _ = grid_bp_batch(grid, weights, log_k_true, X, alpha, delta)
            return m

        def fresh_sample(rng, k):
            return np.stack([prior.sample(rng, N_SITES) for _ in range(k)])

        for cond in cfg["conditions"]:
            if (seed, cond) in done:
                continue
            fresh = cond.startswith("fresh")
            exact = cond.endswith("exact")
            net = BiMessagePassing.init(cfg["hidden"], rng_for("exp33-init", seed, cond))
            snaps = train_sequence_net(
                net, A_fixed, T_SCHEDULE, rng_for("exp33-train-rng", seed, cond),
                checkpoints=cfg["checkpoints"], parameterization=cfg["parameterization"],
                batch_size=cfg["batch_size"], lr=cfg["lr"], grad_clip=cfg["grad_clip"],
                sample_fn=fresh_sample if fresh else None,
                exact_target_fn=exact_target if exact else None,
            )
            val_curve = {
                s: _risk(_region_pairs(
                    net, grid, weights, val,
                    lambda X, t, s=s: posterior_mean(net, snaps[s], X, t, cfg["parameterization"]),
                ))
                for s in sorted(snaps)
            }
            best = min(val_curve, key=val_curve.get)
            test_pairs = _region_pairs(
                net, grid, weights, test,
                lambda X, t: posterior_mean(net, snaps[best], X, t, cfg["parameterization"]),
            )
            row = {
                "seed": seed, "condition": cond, "checkpoint": best,
                "at_cap": int(best == max(snaps)),
                "test_risk": _risk(test_pairs),
                "val_risk": val_curve[best],
            }
            rows.append(row)
            print(f"  seed={seed} {cond:12} ck={best:>6}"
                  f"{' CAP' if row['at_cap'] else ''} test_risk={row['test_risk']:.4f}",
                  flush=True)
            write_csv(dest, rows)
    write_csv(dest, rows)
    return rows


def part_summary(cfg, out):
    import csv as _csv

    src = out / "target_diagnostic.csv"
    if not src.exists():
        raise SystemExit(f"part_summary needs {src} -- run part_run first")
    rows = list(_csv.DictReader(src.open()))
    conds = cfg["conditions"]

    print("\nMean test risk by condition (lower = better; n=%d seeds):" % cfg["seeds"])
    means = {}
    for c in conds:
        vals = [float(r["test_risk"]) for r in rows if r["condition"] == c]
        means[c] = float(np.mean(vals)) if vals else float("nan")
        print(f"  {c:12} {means[c]:.4f}  (n={len(vals)})")

    if all(c in means for c in ("fixed_dsm", "fresh_dsm", "fixed_exact", "fresh_exact")):
        print("\nDecomposition (each vs fixed_dsm, the ordinary-training baseline):")
        data_effect = means["fresh_dsm"] - means["fixed_dsm"]
        target_effect = means["fixed_exact"] - means["fixed_dsm"]
        both_effect = means["fresh_exact"] - means["fixed_dsm"]
        print(f"  removing finite-data effect only : {data_effect:+.4f}")
        print(f"  removing DSM-target variance only : {target_effect:+.4f}")
        print(f"  removing both                     : {both_effect:+.4f}")
        print(f"\n  Residual at fresh_exact ({means['fresh_exact']:.4f}) is the floor "
              f"attributable to architecture/optimisation against the population "
              f"objective -- the only cell that licenses a statement resembling "
              f"'the architecture cannot represent this'.")
    return means


def main() -> None:
    parser = experiment_parser(
        "exp_33_target_diagnostic",
        "2x2 mechanism check: finite data vs DSM target variance (review Phase 2).",
    )
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {
        "run": ("train all four conditions per seed", part_run),
        "summary": ("decompose the residual", part_summary),
    }
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
