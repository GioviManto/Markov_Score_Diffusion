"""Experiment 27 -- when has EM actually converged? The full slow-coordinate trace.

The convergence claims in this project have moved more than any others, and the
reason is that no experiment ever recorded the thing they were about. `exp_18
--parts emtrace` logs `rho_hat` and the likelihood at a hardcoded 60 iterations.
Every statement about the innovation *shape* -- "0.84 at 30 iterations, 2.30 at
120" -- came from ad-hoc scripts, at one draw, at iteration counts chosen after
the fact. That is how a claim gets revised three times.

So this part records, at every evaluated state, the whole diagnostic set needed
to certify (or refuse to certify) convergence:

    observed log-likelihood, and Q before/after the M-step
    rho, and the corrected rho score |dL/drho| from Fisher's identity
    innovation mean, variance, excess kurtosis
    min component sd / grid spacing        -- is a component narrower than the grid?
    min and max component mass             -- has a component died?
    effective number of components         -- exp(entropy of pi)
    label-aligned max parameter change     -- sorted by mu, so relabelling is not motion
    wall-clock seconds, monotonicity flag

Two design points matter.

*Label alignment.* Mixture components are exchangeable, so a raw max-parameter-
change diagnostic reports permutation as non-convergence. Components are sorted
by fitted mean before differencing.

*The rho score uses the analytic gradient*, which carried a sign error until
this branch (see `MixtureInnovationKernel.grad_log_transition_matrix`). A
fixed-point residual near zero is only evidence of a stationary point if that
derivative is right, so this experiment is downstream of that fix.

The sweep is over (N, C, noise design), one array task per cell, because the
question "is 120 enough" has a different answer in each.

    python3 experiments/exp_27_shape_convergence.py --list-parts
    python3 experiments/exp_27_shape_convergence.py --only trace \
        --set 'sizes=(512,)' --set 'components=(8,)' --set n_updates=800
"""

from __future__ import annotations

import time

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import make_grid
from src.em import e_step_multi, q_gradient, q_value
from src.kernels import MixtureInnovationKernel
from src.protocols import one_view_groups as noisy_groups
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites
RHO_TRUE = FROZEN.rho
GRID_A = FROZEN.half_width
GRID_M = FROZEN.n_grid

# Three designs, because information about the innovation shape is not spread
# evenly over the schedule: low t keeps shape information, high t destroys it
# first. A stopping rule tuned on one need not hold on the others.
NOISE_DESIGNS = {
    "low": (0.1, 0.2),
    "uniform": (0.1, 0.2, 0.4, 0.8, 1.6),
    "high": (0.8, 1.6),
}




def _aligned(kernel):
    """Components sorted by fitted mean, so relabelling is not read as motion."""
    order = np.argsort(kernel.mu)
    return np.concatenate([
        [kernel.rho], kernel.pi[order], kernel.mu[order], kernel.s2[order]
    ])


def _diagnostics(kernel, grid, dx):
    """Everything about the fitted kernel that a stopping rule might need."""
    mom = kernel.innovation_moments
    pi = np.asarray(kernel.pi, dtype=float)
    sd = np.sqrt(np.asarray(kernel.s2, dtype=float))
    safe = np.clip(pi, 1e-300, None)
    return {
        "rho": float(kernel.rho),
        "innovation_mean": mom["innovation_mean"],
        "innovation_var": mom["innovation_var"],
        "innovation_excess_kurtosis": mom["innovation_excess_kurtosis"],
        # Below ~1 the component is narrower than the quadrature can resolve, so
        # its shape contribution is a grid artefact rather than a density.
        "min_sd_over_dx": float(sd.min() / dx),
        "min_component_mass": float(pi.min()),
        "max_component_mass": float(pi.max()),
        "effective_components": float(np.exp(-np.sum(safe * np.log(safe)))),
    }


def trace_one(n_chains, n_comp, design, seed, n_updates, grid, weights, dx):
    """One EM run, fully instrumented. Returns a row per evaluated state."""
    prior = LaplaceAR1(RHO_TRUE)
    rng = rng_for("exp27", "data", n_chains, design, seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = noisy_groups(A, NOISE_DESIGNS[design], rng)

    kernel = MixtureInnovationKernel.init(
        n_comp, rho=0.3, var=0.8, rng=rng_for("exp27", "init", n_comp, seed)
    )

    rows = []
    prev_theta = _aligned(kernel)
    prev_ll = -np.inf
    for update in range(n_updates + 1):
        t0 = time.perf_counter()
        log_k = kernel.log_transition_matrix(grid)
        stats = e_step_multi(grid, weights, log_k, groups)

        q_before = q_value(stats, log_k)
        # Fisher's identity: at the parameter Xi was computed at, this is the
        # gradient of the exact marginal log-likelihood.
        score_rho = float(
            q_gradient(stats, kernel.grad_log_transition_matrix(grid))[0]
        ) / stats.n_edges

        theta = _aligned(kernel)
        row = {
            "n_chains": n_chains, "n_components": n_comp, "design": design,
            "seed": seed, "update": update,
            "log_evidence": stats.log_evidence,
            "q_before": q_before,
            "rho_score": score_rho,
            "abs_rho_score": abs(score_rho),
            "max_param_change": float(np.max(np.abs(theta - prev_theta))),
            "rel_ll_increment": (
                float((stats.log_evidence - prev_ll) / abs(prev_ll))
                if np.isfinite(prev_ll) else float("nan")
            ),
            "ll_increment": (
                float(stats.log_evidence - prev_ll) if np.isfinite(prev_ll)
                else float("nan")
            ),
            "monotone_ok": int(
                (not np.isfinite(prev_ll)) or stats.log_evidence >= prev_ll - 1e-8
            ),
            **_diagnostics(kernel, grid, dx),
        }

        if update < n_updates:
            proposal = kernel.m_step(stats, grid)
            row["q_after"] = q_value(stats, proposal.log_transition_matrix(grid))
        else:
            proposal = kernel
            row["q_after"] = float("nan")
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)

        prev_theta, prev_ll, kernel = theta, stats.log_evidence, proposal

    return rows


def part_trace(cfg, grid, weights, out):
    dx = float(grid[1] - grid[0])
    rows = []
    for n_chains in cfg["sizes"]:
        for n_comp in cfg["components"]:
            for design in cfg["designs"]:
                for seed in range(cfg["n_seeds"]):
                    t0 = time.perf_counter()
                    rows += trace_one(n_chains, n_comp, design, seed,
                                      cfg["n_updates"], grid, weights, dx)
                    print(f"  N={n_chains} C={n_comp} {design} seed={seed} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return rows


def part_summary(rows, cfg):
    """First update at which each coordinate is within tolerance of its final value.

    Reported per coordinate rather than as one number, because that separation is
    the finding: rho reaches tolerance long before the shape does, so a rule read
    off rho certifies a kernel whose innovation law is still moving.
    """
    import collections

    tols = {"rho": 1e-3, "innovation_var": 1e-2, "innovation_excess_kurtosis": 2e-2}
    by_run = collections.defaultdict(list)
    for r in rows:
        by_run[(r["n_chains"], r["n_components"], r["design"], r["seed"])].append(r)

    out_rows = []
    for key, rs in sorted(by_run.items()):
        rs = sorted(rs, key=lambda r: r["update"])
        rec = dict(zip(("n_chains", "n_components", "design", "seed"), key))
        rec["n_updates"] = rs[-1]["update"]
        rec["final_log_evidence"] = rs[-1]["log_evidence"]
        rec["monotone_violations"] = sum(1 for r in rs if not r["monotone_ok"])
        for coord, tol in tols.items():
            final = rs[-1][coord]
            scale = max(abs(final), 1e-12)
            late = [r["update"] for r in rs
                    if abs(r[coord] - final) / scale >= tol]
            rec[f"settle_{coord}"] = (max(late) + 1) if late else 0
        for coord in ("innovation_excess_kurtosis", "innovation_var", "rho"):
            rec[f"final_{coord}"] = rs[-1][coord]
        rec["final_abs_rho_score"] = rs[-1]["abs_rho_score"]
        rec["final_min_sd_over_dx"] = rs[-1]["min_sd_over_dx"]
        rec["final_min_component_mass"] = rs[-1]["min_component_mass"]
        rec["final_effective_components"] = rs[-1]["final_effective_components"] \
            if "final_effective_components" in rs[-1] else rs[-1]["effective_components"]
        out_rows.append(rec)
    return out_rows


def main() -> None:
    parser = experiment_parser(
        "exp_27_shape_convergence",
        "Full slow-coordinate traces: when has EM actually converged?",
    )
    args = parser.parse_args()

    quick = {
        "sizes": (256,), "components": (4,), "designs": ("uniform",),
        "n_seeds": 1, "n_updates": 40, "grid_size": 201,
    }
    full = {
        "sizes": (512, 2048), "components": (4, 8, 16),
        "designs": ("low", "uniform", "high"),
        # Was 8, set locally. The frozen count is the one every other
        # experiment reports against, and a convergence curve read off eight
        # draws is exactly the kind of thing this project has had to withdraw.
        "n_seeds": FROZEN.n_seeds, "n_updates": 800, "grid_size": GRID_M,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    def p_trace(grid, weights, out):
        rows = part_trace(cfg, grid, weights, out)
        write_csv(out / "shape_trace.csv", rows)
        write_csv(out / "shape_settle.csv", part_summary(rows, cfg))

    parts = {"trace": ("instrumented EM traces and settling summary", p_trace)}
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, cfg["grid_size"])

    tag = "_".join(selected) if args.only else "all"
    # Array tasks vary sizes/components/designs, so the tag must carry them or
    # concurrent tasks overwrite one another's manifest.
    cell = f"{'-'.join(map(str, cfg['sizes']))}_C{'-'.join(map(str, cfg['components']))}" \
           f"_{'-'.join(cfg['designs'])}"
    write_json(out / f"params_{tag}_{cell}.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "grid_half_width": GRID_A,
        "noise_designs": {k: list(v) for k, v in NOISE_DESIGNS.items()},
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(grid, weights, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
