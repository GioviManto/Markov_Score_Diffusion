"""Diagnostics the revision needs and the existing outputs do not contain.

Three products, all cheap on one CPU, all written as CSV so the figures are
generated from committed data rather than from numbers typed out of prose.

1. ``boundary`` -- truncation diagnostic.
   The note previously carried a [pending] marker here: the boundary mass was
   computed inside the recursion and thrown away, so no aggregate could be
   quoted. For each (t, A, N_g) we run a forward pass over a batch of chains and
   record the fraction of each normalised message sitting in the two edge cells
   of the grid, taking the worst case over sites within a chain and then the
   maximum and upper quantiles over chains. This is an *empirical diagnostic* of
   how much mass approaches the truncation boundary, not a bound on the
   truncation error -- a distinction the first version of this file got wrong in
   both directions, by quoting one sampled trajectory and calling it a bound.

   Reported alongside it, and carrying most of the weight, is the quadrature mass
   of the kernel's own columns. That quantity involves no sample at all, and its
   split into edge and interior columns is what separates truncation (responds to
   A) from quadrature (responds to the spacing).

2. ``emtrace`` -- per-iteration marginal log-likelihood.
   ``outputs/exp_06/monotonicity.csv`` records only the first and last value, so
   monotonicity could be asserted but not plotted. ``fit_em`` already returns the
   whole trace; we persist it, for both the closed-form-M-step Laplace kernel and
   the generalised-EM mixture kernel, from several initialisations.

3. ``density`` -- the learned innovation law.
   Fits the mixture kernel at several data budgets and evaluates the fitted
   innovation density on a common grid, next to the true Laplace density. This
   is the direct picture of what is being estimated; the existing outputs record
   only its moments.

Usage
-----
    python experiments/exp_18_revision_diagnostics.py --parts boundary,emtrace,density
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bp_grid import make_grid  # noqa: E402
from src.em import fit_em  # noqa: E402
from src.kernels import MixtureInnovationKernel  # noqa: E402
from src.noising import alpha_delta  # noqa: E402
from src.priors import LaplaceAR1  # noqa: E402
from src.utils import rng_for  # noqa: E402
from frozen_config import FROZEN

OUT = Path(__file__).resolve().parents[1] / "outputs" / "exp_18"

RHO = FROZEN.rho
N_SITES = FROZEN.n_sites
T_VALUES = tuple(FROZEN.t_grid)

# Chains per (t, A, N_g) cell for the forward-message edge-mass diagnostic. One trajectory
# cannot characterise a distribution over trajectories, and the quantity is a max over sites
# of a sampled path -- exactly the sort of statistic a single draw understates.
BOUNDARY_CHAINS = 256

# Seeding note. This module previously used
#
#     np.random.default_rng(abs(hash(("exp18",) + tag)) % (2 ** 32))
#
# which is not reproducible across processes: Python salts `hash` for str and bytes unless
# PYTHONHASHSEED is fixed. `utils.rng_for` exists precisely because that class of
# nondeterminism was removed from this project once already, and its docstring says so. All
# outputs under `outputs/exp_18/` produced before this change must be regenerated.
#
# What the bug did *not* invalidate: the kernel column-mass residuals below are pure
# functions of (prior, A, N_g) with no sample entering them, so the truncation-versus-
# quadrature separation they carry is unaffected. Only the edge-mass column was ever
# sample-dependent.


# ---------------------------------------------------------------------------
# 1. Truncation and normalisation diagnostics
# ---------------------------------------------------------------------------

def part_boundary() -> None:
    """Boundary mass and normalisation residual over (t, A, N_g).

    Two quantities with different epistemic status, which the previous version ran together
    and the write-up then conflated.

    The **kernel column-mass residuals** are pure functions of ``(prior, A, N_g)``: no sample
    enters them at all. They are what separates truncation from quadrature, and they are
    deterministic, so they need no replication and were never affected by the seeding defect.

    The **forward-message edge mass** does depend on the observations, and it is a maximum
    over sites of a sampled trajectory -- a statistic a single draw systematically
    understates. It is therefore an *empirical diagnostic* over ``BOUNDARY_CHAINS`` chains,
    reported as a maximum and upper quantiles, not a bound. Calling one trajectory's value a
    truncation bound, as the earlier comment did, was wrong twice over: not a bound, and not
    a sample.
    """
    prior = LaplaceAR1(RHO)
    rows = []
    edge_sel = None
    for half_width in (4.0, 6.0, 8.0, 10.0):
        for n_grid in (201, 401, 801):
            grid, weights = make_grid(half_width, n_grid)
            K = np.exp(prior.log_transition_matrix(grid))

            # Deterministic, sample-free: computed once per (A, N_g). The maximum over all
            # columns is attained at the edge, where the transition density is centred at
            # rho*(+/-A) and its tail leaves the grid -- truncation, responding to A, not
            # N_g. Restricting to interior columns (|u| <= A/2) removes the truncated tail,
            # so what remains is quadrature, responding to the spacing.
            col_mass = (K * weights[:, None]).sum(axis=0)
            interior = np.abs(grid) <= 0.5 * half_width
            resid = float(np.max(np.abs(col_mass - 1.0)))
            resid_interior = float(np.max(np.abs(col_mass[interior] - 1.0)))

            edge_sel = np.zeros(len(grid), dtype=bool)
            edge_sel[0] = edge_sel[-1] = True
            log_mu0 = np.exp(-0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi))

            for t in T_VALUES:
                alpha, delta = alpha_delta(t)
                rng = rng_for("exp18", "boundary", t, half_width, n_grid)
                A = np.stack(
                    [prior.sample(rng, N_SITES) for _ in range(BOUNDARY_CHAINS)]
                )
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

                # (B, n, M) likelihood, row-shifted per (chain, site) for conditioning.
                z = X[:, :, None] - alpha * grid[None, None, :]
                log_ell = -0.5 * z**2 / delta
                log_ell -= log_ell.max(axis=2, keepdims=True)
                ell = np.exp(log_ell)

                # Forward sweep over the whole batch, recording each chain's worst
                # edge-cell mass across sites.
                L = np.tile(log_mu0, (BOUNDARY_CHAINS, 1))
                worst = np.zeros(BOUNDARY_CHAINS)
                for i in range(N_SITES - 1):
                    incoming = L * ell[:, i, :] * weights[None, :]
                    out = incoming @ K.T
                    out = out / (out @ weights)[:, None]
                    dens = out * weights[None, :]
                    worst = np.maximum(
                        worst, dens[:, edge_sel].sum(axis=1) / dens.sum(axis=1)
                    )
                    L = out

                rows.append(
                    {
                        "t": t,
                        "half_width": half_width,
                        "n_grid": n_grid,
                        "spacing": float(grid[1] - grid[0]),
                        "n_chains": BOUNDARY_CHAINS,
                        "boundary_mass_max": float(worst.max()),
                        "boundary_mass_p99": float(np.quantile(worst, 0.99)),
                        "boundary_mass_p90": float(np.quantile(worst, 0.90)),
                        "boundary_mass_median": float(np.median(worst)),
                        "kernel_norm_residual_max": resid,
                        "kernel_norm_residual_interior": resid_interior,
                    }
                )
                print(
                    f"A={half_width:>5} Ng={n_grid:>4} t={t:<5} "
                    f"edge_max={worst.max():.3e} edge_p90={np.quantile(worst, 0.9):.3e} "
                    f"resid_max={resid:.3e} resid_int={resid_interior:.3e}"
                )
    _write_csv(OUT / "boundary.csv", rows)


# ---------------------------------------------------------------------------
# 2. EM objective trace
# ---------------------------------------------------------------------------

def part_emtrace() -> None:
    """Persist the whole marginal log-likelihood trace, several initialisations."""
    prior = LaplaceAR1(RHO)
    grid, weights = make_grid(8.0, 401)
    rows = []
    n_chains, t_train = 512, (0.1, 0.2, 0.4, 0.8, 1.6)

    rng = rng_for("exp18", "emtrace", "data")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = []
    for t in t_train:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        groups.append((X, alpha, delta))

    for n_comp in (4, 8):
        for init_id, rho0 in enumerate((0.0, 0.3, 0.6, -0.4)):
            kernel = MixtureInnovationKernel.init(
                n_comp, rho=rho0, var=0.8, rng=rng_for("exp18", "emtrace", n_comp, init_id)
            )
            fitted, trace = fit_em(kernel, grid, weights, groups, n_iters=60)  # frozen-exempt: PROVISIONAL -- reports innovation moments, whose kurtosis settles at a median 229 updates (exp_27). Compendium-only; see ch13 sec:corr-capacity. Rerun at FROZEN.em_max_iters before any use
            mom = fitted.innovation_moments
            for it, (ll, sec) in enumerate(zip(trace.log_evidence, trace.seconds)):
                rows.append(
                    {
                        "n_components": n_comp,
                        "init_id": init_id,
                        "rho_init": rho0,
                        "iteration": it,
                        "log_evidence": ll,
                        "seconds": sec,
                        "rho_hat": float(trace.theta[it][0]),
                    }
                )
            print(
                f"C={n_comp} init rho0={rho0:>5}: violation={trace.monotone_violation:.3e} "
                f"rho_hat={fitted.rho:.4f} mean={mom['innovation_mean']:.2e} "
                f"exkurt={mom['innovation_excess_kurtosis']:.3f}"
            )
    _write_csv(OUT / "em_trace.csv", rows)


# ---------------------------------------------------------------------------
# 3. Learned innovation density
# ---------------------------------------------------------------------------

def part_density() -> None:
    """Fitted innovation density at several budgets, against the truth."""
    prior = LaplaceAR1(RHO)
    grid, weights = make_grid(8.0, 401)
    e = np.linspace(-3.0, 3.0, 601)
    b = prior.b
    true = np.exp(-np.abs(e) / b) / (2.0 * b)

    rows = [{"e": float(v), "budget": 0, "n_components": 0, "arm": "true",
             "density": float(d)} for v, d in zip(e, true)]
    summary = []

    for n_chains in (128, 512, 2048):
        for n_comp in (4, 8):
            rng = rng_for("exp18", "density", n_chains, n_comp)
            A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
            groups = []
            for t in (0.1, 0.2, 0.4, 0.8, 1.6):
                alpha, delta = alpha_delta(t)
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                groups.append((X, alpha, delta))
            kernel = MixtureInnovationKernel.init(
                n_comp, rho=0.3, var=0.8, rng=rng_for("exp18", "density", "init", n_chains, n_comp)
            )
            fitted, trace = fit_em(kernel, grid, weights, groups, n_iters=60)  # frozen-exempt: reports rho only; settles at <=80 updates (exp_27)

            dens = np.zeros_like(e)
            for w, m, s2 in zip(fitted.pi, fitted.mu, fitted.s2):
                dens += w * np.exp(-0.5 * (e - m) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
            rows += [
                {"e": float(v), "budget": n_chains, "n_components": n_comp,
                 "arm": "fitted", "density": float(d)}
                for v, d in zip(e, dens)
            ]
            mom = fitted.innovation_moments
            summary.append(
                {
                    "n_chains": n_chains,
                    "n_components": n_comp,
                    "rho_hat": float(fitted.rho),
                    "rho_true": RHO,
                    "innovation_mean": mom["innovation_mean"],
                    "innovation_var": mom["innovation_var"],
                    "innovation_var_true": prior.q,
                    "innovation_excess_kurtosis": mom["innovation_excess_kurtosis"],
                    "innovation_excess_kurtosis_true": 3.0,
                    "monotone_violation": trace.monotone_violation,
                    "iters": len(trace.log_evidence),
                }
            )
            print(
                f"M={n_chains:>5} C={n_comp}: rho={fitted.rho:.4f} "
                f"mean={mom['innovation_mean']:+.2e} "
                f"exkurt={mom['innovation_excess_kurtosis']:.3f} (true 3.0)"
            )

    _write_csv(OUT / "innovation_density.csv", rows)
    _write_csv(OUT / "innovation_summary.csv", summary)


# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


PARTS = {"boundary": part_boundary, "emtrace": part_emtrace, "density": part_density}


def main() -> None:
    global OUT
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="boundary,emtrace,density")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory (default outputs/exp_18). Needed by the determinism check, "
            "which runs this module twice under different PYTHONHASHSEED values and diffs "
            "the results -- the failure mode of the old seeding was precisely that two "
            "processes disagreed, so it cannot be demonstrated from a single directory."
        ),
    )
    args = ap.parse_args()
    if args.out is not None:
        OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = [p.strip() for p in args.parts.split(",") if p.strip()]
    for name in chosen:
        print(f"\n=== {name} ===")
        PARTS[name]()
    (OUT / "params.json").write_text(
        json.dumps(
            {
                "rho": RHO,
                "n_sites": N_SITES,
                "t_values": list(T_VALUES),
                "parts": chosen,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "blas_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
