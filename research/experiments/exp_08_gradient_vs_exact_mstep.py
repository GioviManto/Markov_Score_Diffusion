"""Experiment 08 -- The literal gradient route vs the exact M-step.

The suggestion that started this layer was phrased as gradient ascent: "starting
from a random initialization, we iterate following the gradient of the loss
constructed from the likelihood function along the direction of the parameters,
evaluated on a finite number of datapoints". That is exactly what Fisher's
identity makes cheap -- `q_gradient` returns the gradient of the *exact*
marginal log-likelihood from one BP pass, with no differentiation through the
recursion -- so the route is available and worth measuring rather than assumed
inferior.

Both routes share the same E-step and the same statistic Xi. They differ only
in what they do with it:

    gradient ascent   theta <- theta + eta * <Xi, grad log K>      (Fisher)
    exact M-step      theta <- argmax_theta <Xi, log K_theta>      (EM proper)

Measured on the Gaussian kernel, whose M-step is closed form and whose
derivatives are smooth, so the comparison isolates the update rule and not the
quadrature. Reported: iterations to reach a fixed accuracy, final accuracy, and
whether monotonicity survives.

The Laplace kernel is included as the cautionary case: its rho-derivative
carries a sign discontinuity that trapezoidal quadrature integrates poorly, so
the gradient route inherits a bias the exact M-step does not have.
"""

from __future__ import annotations

import numpy as np

from common import experiment_parser, provenance
from src.bp_grid import make_grid
from src.em import e_step_multi, fit_em, q_gradient
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel
from src.plotting import new_figure, save_figure
from src.protocols import one_view_groups as noisy_groups
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites   # was 24
RHO_TRUE = FROZEN.rho   # was 0.8
GRID_A = FROZEN.half_width
T_TRAIN = tuple(FROZEN.t_grid)




def gradient_ascent(kernel, grid, weights, groups, lr, n_iters, project):
    """Fisher-identity gradient ascent on the exact marginal log-likelihood.

    `project` maps a raw parameter vector back into the admissible set (positive
    scale, |rho| < 1); without it the iterate can leave the domain in one step
    and the run dies rather than merely converging badly, which would confuse
    "the learning rate is too large" with "the method does not work".

    Every state in the trace is *evaluated*: the loop records (theta, logL) for
    the same kernel, and the returned kernel is the one whose statistics sit at
    the end of the trace. The previous version appended before updating and then
    returned the post-update iterate, so `param_err` and `logL_final` described
    two different parameters -- the reported likelihood belonged to the
    second-to-last kernel. This yields `n_updates + 1` evaluated states; the
    update count is returned separately rather than inferred from the trace
    length.
    """
    trace = {"log_evidence": [], "theta": []}
    current = kernel

    def evaluate(kern):
        return e_step_multi(
            grid, weights, kern.log_transition_matrix(grid), groups
        )

    stats = evaluate(current)
    trace["log_evidence"].append(stats.log_evidence)
    trace["theta"].append(np.asarray(current.theta, dtype=float).copy())

    for _ in range(n_iters):
        grad = q_gradient(stats, current.grad_log_transition_matrix(grid))
        theta_new = np.asarray(current.theta, dtype=float) + lr * grad / stats.n_edges
        current = project(theta_new)
        stats = evaluate(current)
        trace["log_evidence"].append(stats.log_evidence)
        trace["theta"].append(np.asarray(current.theta, dtype=float).copy())

    trace["n_updates"] = n_iters
    return current, trace


def _monotone_violation(values) -> float:
    d = np.diff(np.asarray(values, dtype=float))
    return float(max(0.0, -d.min())) if d.size else 0.0


def run_family(name, prior, make_kernel, project, truth, grid, weights,
               n_chains, lrs, n_iters):
    rng = rng_for("exp08", name, n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = noisy_groups(A, T_TRAIN, rng)

    rows, traces = [], {}

    fitted, trace = fit_em(make_kernel(0.3, 0.8), grid, weights, groups,
                           n_iters=n_iters, tol=0.0)
    err = np.abs(np.asarray(fitted.theta, dtype=float) - truth)
    traces["exact M-step"] = trace.log_evidence
    rows.append({
        "family": name, "route": "exact_m_step", "lr": np.nan,
        "n_chains": n_chains, "n_evaluated_states": len(trace.log_evidence),
        "n_updates": len(trace.log_evidence) - 1,
        "logL_final": trace.log_evidence[-1],
        "param0_err": float(err[0]), "param1_err": float(err[1]),
        "monotone_violation": trace.monotone_violation,
        "logL_gap_to_em": 0.0,
    })
    best_logL = trace.log_evidence[-1]

    for lr in lrs:
        fitted_g, tr = gradient_ascent(
            make_kernel(0.3, 0.8), grid, weights, groups, lr, n_iters, project
        )
        err = np.abs(np.asarray(fitted_g.theta, dtype=float) - truth)
        traces[f"gradient, $\\eta={lr:g}$"] = tr["log_evidence"]
        rows.append({
            "family": name, "route": "gradient", "lr": lr,
            "n_chains": n_chains,
            "n_evaluated_states": len(tr["log_evidence"]),
            "n_updates": tr["n_updates"],
            "logL_final": tr["log_evidence"][-1],
            "param0_err": float(err[0]), "param1_err": float(err[1]),
            "monotone_violation": _monotone_violation(tr["log_evidence"]),
            "logL_gap_to_em": float(best_logL - tr["log_evidence"][-1]),
        })
    return rows, traces


def main() -> None:
    parser = experiment_parser(
        "exp_08_gradient_vs_exact_mstep",
        "Fisher-identity gradient ascent vs the exact EM M-step.",
    )
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, 201 if args.quick else 301)
    n_chains = 64 if args.quick else 256
    n_iters = 30 if args.quick else 120
    lrs = (0.5, 2.0) if args.quick else (0.1, 0.5, 2.0, 8.0)

    write_json(out / "params.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "grid_size": len(grid),
        "grid_half_width": GRID_A, "t_train": T_TRAIN, "n_chains": n_chains,
        "n_iters": n_iters, "learning_rates": lrs, "quick": args.quick,
        **provenance(),
    })

    def clip_gauss(v):
        return GaussianAR1Kernel(float(np.clip(v[0], -0.99, 0.99)),
                                 float(np.clip(v[1], 1e-3, 10.0)))

    def clip_lap(v):
        return LaplaceAR1Kernel(float(np.clip(v[0], -0.99, 0.99)),
                                float(np.clip(v[1], 1e-3, 10.0)))

    all_rows = []
    print("Gaussian kernel (smooth) ...", flush=True)
    rows_g, traces_g = run_family(
        "gaussian", GaussianAR1(RHO_TRUE), GaussianAR1Kernel, clip_gauss,
        np.array([RHO_TRUE, 1.0 - RHO_TRUE**2]),
        grid, weights, n_chains, lrs, n_iters,
    )
    all_rows += rows_g

    print("Laplace kernel (non-smooth derivative) ...", flush=True)
    lap = LaplaceAR1(RHO_TRUE)
    rows_l, traces_l = run_family(
        "laplace", lap, LaplaceAR1Kernel, clip_lap,
        np.array([RHO_TRUE, lap.b]),
        grid, weights, n_chains, lrs, n_iters,
    )
    all_rows += rows_l

    write_csv(out / "gradient_vs_exact.csv", all_rows)

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    for axis, traces, title in (
        (ax[0], traces_g, "Gaussian kernel"),
        (ax[1], traces_l, "Laplace kernel"),
    ):
        for label, ev in traces.items():
            style = "k-" if label == "exact M-step" else "-"
            axis.plot(ev, style, lw=2.0 if label == "exact M-step" else 1.1,
                      label=label)
        axis.set_xlabel("iteration (one exact BP E-step each)")
        axis.set_ylabel(r"exact $\log p_t(x)$")
        axis.set_title(title)
        axis.legend(fontsize=8)
    save_figure(fig, out / "gradient_vs_exact.png")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
