"""Experiment 32 -- does mixture capacity actually saturate near C=8?

WHY THIS REPLACES THE PAIRED-CAPACITY TABLE IN CHAPTER 11
-----------------------------------------------------------
The thesis reported held-out evidence per edge at C in {2,4,8,16}, paired
against C=1, six seeds, one training size (N=128), one initialisation per
cell. C=8 and C=16 differed by 5e-6, inside one standard error, and an
earlier draft read that as "nothing further is bought past C~8" -- saturation.

Round-two review, §6.3 and §10.6: failing to resolve a difference is not
evidence there is none. Six seeds with no predeclared equivalence region
cannot establish equivalence; it can only fail to reject a null it was never
positioned to test. The thesis text was corrected to say exactly that
(Chapter 11, "failure to resolve" rather than "saturation") -- this experiment
is what would let the stronger claim be made honestly, if the data supports it.

WHAT THIS DOES DIFFERENTLY
---------------------------
  * Sixteen paired seeds, not six -- the same replication count as the
    headline table, for the same reason: three seeds read a M^{-1/2} decay as
    flat, and six is not obviously enough either.
  * THREE initialisations per (seed, size, C) cell, and the one used for
    reporting is the one with the best VALIDATION held-out evidence -- so the
    contrast reflects what a real fit would select, not a lucky or unlucky
    single draw. (Configurable via `inits_per_cell`.)
  * Separate validation and test bundles per seed, matching exp_31's
    discipline: validation picks the initialisation, test is what gets
    reported, and the two never share a chain.
  * Three metrics, not one: held-out log-evidence per edge (what the withdrawn
    claim used), schedule-level score risk (the estimand exp_31 standardises
    on -- summed numerator and denominator, square-rooted once), and the
    parent-law-weighted Hellinger-squared from `src/metrics.py` (round-two
    review, "Hellinger" section -- an intrinsic distance under the shared
    parent law, not an unweighted vote per grid column).
  * A PREDECLARED equivalence region for the C=16 vs C=8 contrast, checked
    before the numbers are seen, exactly as the review's §10.6 specifies:
        |mean paired Delta(log-evidence/edge)| < 1e-4
        |R_16 - R_8| / R_8 < 1%          (schedule-level score risk)
    "Capacity saturates by eight" is supported only if a paired bootstrap CI
    for BOTH contrasts lies entirely inside its region. Anything else is
    reported as what it is: a resolved difference, or an unresolved question.

EM iteration budget. The withdrawn shape-based capacity claims used a fixed
40 iterations, and Section~sec:em-convergence separately measured that the
fitted SHAPE can need ~2000 to settle while the evidence and rho settle by
~25-40. Evidence-based conclusions are not the ones that confound was about --
but "not the same confound" turned out not to mean "immune to any confound".

CORRECTION, after the first run at em_cap=400. That run reported C=16 with
measurably lower held-out evidence and higher score risk than C=8, resolved
(not just unresolved) against the predeclared equivalence region -- and 0% of
cells satisfied the formal tol=1e-9 criterion at either capacity. Checking the
evidence trace directly: at iteration 400, C=8's gain over the last 100
iterations was 0.025 nats (per-edge gain 9.9e-9, a hair past tolerance,
practically flat) while C=16's was 0.77 nats over the same window (per-edge
gain 6.9e-8, thirty times larger). C=16 was still climbing substantially when
the cap stopped it; C=8 was not. The "C=16 is worse" result at em_cap=400 is
therefore confounded with C=16 being less converged, which is exactly the
capacity-vs-convergence-rate confound this experiment exists to avoid, arrived
at from a different direction than the withdrawn shape claim. `em_cap` is
raised accordingly, sized from measuring how long C=16 actually takes to reach
a comparably tight per-edge gain (see the commit message / job log for the
number used).
"""

from __future__ import annotations

import numpy as np

from common import (
    apply_overrides,
    experiment_parser,
    provenance,
    resolved_config_hash,
    select_parts,
)
from frozen_config import FROZEN
from src.bp_grid import grid_bp_batch, make_grid
from src.em import e_step_multi, fit_em
from src.kernels import MixtureInnovationKernel
from src.metrics import transition_hellinger
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = FROZEN.n_sites
RHO = FROZEN.rho
GRID_M = FROZEN.n_grid
GRID_A = FROZEN.half_width
T_SCHEDULE = FROZEN.t_grid

SETTINGS = dict(
    rho=RHO,
    grid_size=GRID_M,
    half_width=GRID_A,
    components=(1, 2, 4, 8, 16),
    # Two sizes rather than the review's three: capacity questions are most
    # sample-starved (and most interesting) at N=128/512; N=2048 is where
    # capacity is least likely to bind and is dropped to keep the sweep
    # tractable at sixteen paired seeds. Disclosed, not silent -- see the
    # module docstring.
    sizes=(128, 512),
    seeds=16,
    seed0=0,
    # Contiguous shard of the seed range this process is responsible for, so a
    # cluster array can split by seed without an undeclared override -- the
    # defect that made exp_12_scaled's provenance unrecoverable. Declared here
    # rather than accepted as a bare `--set eff_seed0=...` outside the schema.
    shard_seed0=0,
    shard_seeds=16,
    inits_per_cell=2,
    n_val=512,
    n_test=1024,
    # Raised from 400 after the first run showed C=16 still moving 30x faster
    # than C=8 in the last 100 iterations before the old cap -- see the module
    # docstring's correction note and outputs/README_exp32_capacity.md. 1200
    # matches this project's established "generous EM cap" (exp_31's and
    # exp_34's checkpoint ladders both top out here) and is independently
    # confirmed by direct measurement: a representative C=16 fit run to 2000
    # iterations shows gain/edge falling from 6.9e-8 at 400 to ~6e-9 by 800,
    # then PLATEAUING there through 2000 (6.08e-9 @1200, 5.92e-9 @1600,
    # 6.09e-9 @2000) -- it never crosses the strict tol=1e-9 threshold, but
    # 800-1200 already captures essentially all of the real movement. So
    # `em_converged` will likely still read False for C=16 even here, and
    # that is expected: this cap targets the practical plateau, not the
    # formal tolerance, which measurement shows this fit will not reach at
    # any practical iteration count. fit_em stops on tol regardless of the
    # cap either way, so cells that already converged by 400 (the lower
    # capacities) are unaffected; only the slow ones get the runway they
    # needed to reach THEIR plateau.
    em_cap=1200,
    # Equivalence region, predeclared (round-two review §10.6). Do not tune
    # these after seeing the contrast; that would defeat the point of
    # predeclaring them.
    equiv_logev_abs=1e-4,
    equiv_risk_rel=0.01,
    boot_resamples=20_000,
)

QUICK = dict(
    sizes=(128,),
    seeds=2,
    shard_seeds=2,
    inits_per_cell=1,
    n_val=64,
    n_test=64,
    em_cap=20,
    grid_size=201,
    components=(1, 2, 4),
    boot_resamples=500,
)


def _seed_range(cfg):
    lo = cfg["seed0"] + cfg["shard_seed0"]
    return range(lo, lo + cfg["shard_seeds"])


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


def _schedule_risk_pairs(kernel, grid, weights, bundle):
    """(sq_err, sq_ref) per noise level -- exp_31's estimand, not duplicated."""
    log_k = kernel.log_transition_matrix(grid)
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        alpha, delta = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        s_hat = -(X - alpha * m) / delta
        s_ref = -(X - alpha * m_ref) / delta
        pairs.append((float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())))
    return pairs


def _risk(pairs) -> float:
    num = sum(e for e, _ in pairs)
    den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def part_sweep(cfg, out):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    log_k_true = prior.log_transition_matrix(grid)

    dest = out / "capacity_equivalence.csv"
    rows, done = [], set()
    if dest.exists():
        import csv as _csv
        rows = list(_csv.DictReader(dest.open()))
        done = {(int(r["seed"]), int(r["n_chains"]), int(r["n_components"])) for r in rows}
        print(f"resuming: {len(done)} cell(s) already on disk", flush=True)

    for seed in _seed_range(cfg):
        pending = [(n, c) for n in cfg["sizes"] for c in cfg["components"]
                   if (seed, n, c) not in done]
        if not pending:
            continue
        _, val = _bundle(prior, grid, weights, "exp32-val", seed, cfg["n_val"], T_SCHEDULE)
        _, test = _bundle(prior, grid, weights, "exp32-test", seed, cfg["n_test"], T_SCHEDULE)

        for n_chains in sorted({n for n, _ in pending}):
            A = np.stack([
                prior.sample(rng_for("exp32-train", seed, n_chains), N_SITES)
                for _ in range(n_chains)
            ])
            # One-view split across the schedule, matching exp_31 and the
            # headline protocol: each chain is noised at exactly one level.
            rng_split = rng_for("exp32-split", seed, n_chains)
            idx = np.array_split(rng_split.permutation(n_chains), len(T_SCHEDULE))
            noise = rng_for("exp32-noise", seed, n_chains)
            groups = []
            for t, ix in zip(T_SCHEDULE, idx):
                al, de = alpha_delta(t)
                sub = A[ix]
                groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))

            for _, n_comp in [(n, c) for n, c in pending if n == n_chains]:
                best_kernel, best_val_ev = None, -np.inf
                for j in range(cfg["inits_per_cell"]):
                    kernel, trace = fit_em(
                        MixtureInnovationKernel.init(
                            n_comp, rho=0.3, var=0.8,
                            rng=rng_for("exp32-init", seed, n_chains, n_comp, j),
                        ),
                        grid, weights, groups,
                        n_iters=cfg["em_cap"], tol=FROZEN.em_loglik_tol,
                    )
                    log_k = kernel.log_transition_matrix(grid)
                    val_groups = [(val[t][0], *alpha_delta(t)) for t in T_SCHEDULE]
                    held_val = e_step_multi(grid, weights, log_k, val_groups)
                    val_ev = held_val.log_evidence / held_val.n_edges
                    if val_ev > best_val_ev:
                        best_val_ev, best_kernel, best_trace = val_ev, kernel, trace

                log_k = best_kernel.log_transition_matrix(grid)
                test_groups = [(test[t][0], *alpha_delta(t)) for t in T_SCHEDULE]
                held_test = e_step_multi(grid, weights, log_k, test_groups)

                risk_pairs = _schedule_risk_pairs(best_kernel, grid, weights, test)
                hell = transition_hellinger(log_k, log_k_true, grid, weights)
                scale = best_kernel.scale_diagnostics(grid)

                rows.append({
                    "seed": seed, "n_chains": n_chains, "n_components": n_comp,
                    "inits_tried": cfg["inits_per_cell"],
                    "val_log_evidence_per_edge": best_val_ev,
                    "test_log_evidence_per_edge": held_test.log_evidence / held_test.n_edges,
                    "schedule_score_risk": _risk(risk_pairs),
                    "hellinger_weighted_mean_sq": hell["hellinger_weighted_mean_sq"],
                    "hellinger_weighted_mean": hell["hellinger_weighted_mean"],
                    "s_min_over_h": scale.get("s_min_over_h"),
                    "effective_n_components": scale.get("effective_n_components"),
                    "em_iters_used": len(best_trace.log_evidence),
                    "em_converged": best_trace.converged,
                })
                print(f"  seed={seed} n={n_chains} C={n_comp:2d} "
                      f"val_ev/edge={best_val_ev:+.6f} "
                      f"risk={rows[-1]['schedule_score_risk']:.4f} "
                      f"hell2={hell['hellinger_weighted_mean_sq']:.2e} "
                      f"s_min/h={scale.get('s_min_over_h', float('nan')):.2f} "
                      f"iters={rows[-1]['em_iters_used']}"
                      f"{'' if best_trace.converged else ' CENSORED'}", flush=True)
                write_csv(dest, rows)
    write_csv(dest, rows)
    return rows


def part_contrast(cfg, out):
    """The predeclared C=16 vs C=8 equivalence test, plus every adjacent pair."""
    import csv as _csv

    src = out / "capacity_equivalence.csv"
    if not src.exists():
        raise SystemExit(f"part_contrast needs {src} -- run part_sweep first")
    rows = list(_csv.DictReader(src.open()))
    comps = sorted({int(r["n_components"]) for r in rows})
    sizes = sorted({int(r["n_chains"]) for r in rows})
    rng = np.random.default_rng(20260824)

    def _get(n_chains, c, metric):
        by_seed = {}
        for r in rows:
            if int(r["n_chains"]) == n_chains and int(r["n_components"]) == c:
                by_seed[int(r["seed"])] = float(r[metric])
        return by_seed

    contrasts = []
    for n_chains in sizes:
        for i in range(len(comps) - 1):
            lo, hi = comps[i], comps[i + 1]
            for metric in ("test_log_evidence_per_edge", "schedule_score_risk"):
                a, b = _get(n_chains, lo, metric), _get(n_chains, hi, metric)
                shared = sorted(set(a) & set(b))
                if not shared:
                    continue
                diffs = np.array([b[s] - a[s] for s in shared])
                boot = rng.choice(diffs, size=(cfg["boot_resamples"], len(diffs)), replace=True)
                boot_means = boot.mean(axis=1)
                lo_ci, hi_ci = np.percentile(boot_means, [2.5, 97.5])
                mean_diff = float(diffs.mean())

                if metric == "test_log_evidence_per_edge":
                    region = cfg["equiv_logev_abs"]
                    inside = abs(lo_ci) < region and abs(hi_ci) < region
                else:
                    base = float(np.mean([a[s] for s in shared]))
                    region = cfg["equiv_risk_rel"] * abs(base)
                    inside = abs(lo_ci) < region and abs(hi_ci) < region

                contrasts.append({
                    "n_chains": n_chains, "c_lo": lo, "c_hi": hi, "metric": metric,
                    "n_pairs": len(shared), "mean_diff": mean_diff,
                    "ci_lo": float(lo_ci), "ci_hi": float(hi_ci),
                    "equivalence_region": region,
                    "ci_entirely_inside_region": bool(inside),
                })

    write_csv(out / "capacity_contrasts.csv", contrasts)

    print("\nPredeclared equivalence check, C=16 vs C=8:")
    for n_chains in sizes:
        rel = [c for c in contrasts if c["n_chains"] == n_chains
               and c["c_lo"] == 8 and c["c_hi"] == 16]
        if not rel:
            print(f"  N={n_chains}: C=8 and C=16 not both present")
            continue
        both_inside = all(c["ci_entirely_inside_region"] for c in rel)
        verdict = "EQUIVALENT (supports 'saturates by 8')" if both_inside \
            else "NOT established as equivalent -- resolved difference or unresolved"
        print(f"  N={n_chains}: {verdict}")
        for c in rel:
            print(f"    {c['metric']:28} diff={c['mean_diff']:+.3e}  "
                  f"CI=[{c['ci_lo']:+.3e}, {c['ci_hi']:+.3e}]  "
                  f"region=+-{c['equivalence_region']:.3e}  "
                  f"inside={c['ci_entirely_inside_region']}")
    return contrasts


def main() -> None:
    parser = experiment_parser(
        "exp_32_capacity_equivalence",
        "Does held-out evidence actually stop distinguishing capacities by C=8?",
    )
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {
        "sweep": ("fit every (seed, size, capacity) cell", part_sweep),
        "contrast": ("predeclared C=16 vs C=8 equivalence test", part_contrast),
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
