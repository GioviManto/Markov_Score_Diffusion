"""Experiment 22 -- does the channel really destroy SHAPE information fastest?

The claim under test
--------------------
The note states that the OU channel destroys higher-order structure faster than second-order
structure, on the strength of `exp_06`: Fisher information for the innovation variance ``q``
falls 142x between t = 0.05 and t = 1.6 while information for the correlation ``rho`` falls
only 26x.

An external review showed this does not support the claim, and it is right. That experiment
runs in a **Gaussian** AR(1) with parameters ``(rho, q)``. Both are second-order: ``q`` is a
scale, not a shape. What was measured is that variance information decays faster than
correlation information -- interesting, but silent about higher-order structure, because the
model has no higher-order parameter to be informative about.

Two further problems the review identified, both fixed here:

  * ``g.T @ g / N`` is an empirical **outer-product** Fisher estimate, not observed
    information. Named accordingly below.
  * Comparing diagonal entries ignores nuisance coupling. If estimating shape requires also
    estimating ``rho`` and ``q``, the relevant quantity is the *efficient* information after
    projecting the nuisance directions out.

What this experiment does instead
---------------------------------
`src/shape_kernel.GeneralizedGaussianKernel` carries a genuine shape coordinate:

    p(e) propto exp( -(|e|/s)^beta ),   s(q, beta) chosen so Var(e) = q exactly.

``beta = 1`` is Laplace, ``beta = 2`` is Gaussian, and every member has the same variance, so
``beta`` moves the fourth moment while leaving the second fixed -- exactly the separation the
claim requires. Excess kurtosis runs 3 -> 0 across that range.

At each noise level we form the full 3x3 matrix for ``theta = (rho, q, beta)`` and report,
for each coordinate, both

    J[x, x]                          the raw diagonal, comparable to what exp_06 reported, and
    I_{xx . eta} = 1 / (J^{-1})[x, x]  the efficient information with the other two projected out,

since the second is what bounds the variance of any estimator that must also fit the
nuisance parameters. The difference between the two is the size of the mistake the old
comparison was making.

The headline number is the decay ratio of each coordinate's efficient information across the
schedule. Only if shape decays faster than correlation does the original sentence survive --
in a corrected form, and about a parameter that is actually higher-order.

This experiment deliberately does not fit anything. The information is evaluated at the truth
by Fisher's identity, one BP pass per chain, so nothing here depends on an optimiser.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import make_grid
from src.em import e_step_multi, q_gradient
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.shape_kernel import (
    GeneralizedGaussianKernel,
    generalized_gaussian_excess_kurtosis,
)
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites
RHO_TRUE = FROZEN.rho
COORDS = ("rho", "q", "beta")


def sample_generalized_gaussian(
    rng: np.random.Generator, kernel: GeneralizedGaussianKernel, n: int
) -> np.ndarray:
    """One chain with generalised-Gaussian innovations of variance exactly ``q``.

    A generalised Gaussian of shape ``beta`` and scale ``s`` is drawn as
    ``sign * s * G^(1/beta)`` with ``G ~ Gamma(1/beta, 1)``: the standard construction, and
    the only sampler here, so `tests/test_shape_kernel.py` checks the variance it produces
    against the closed form rather than against itself.
    """
    s = kernel.scale
    a = np.empty(n)
    a[0] = rng.standard_normal()
    g = rng.gamma(1.0 / kernel.beta, 1.0, size=n - 1)
    eps = rng.choice([-1.0, 1.0], size=n - 1) * s * g ** (1.0 / kernel.beta)
    for i in range(1, n):
        a[i] = kernel.rho * a[i - 1] + eps[i - 1]
    return a


def outer_product_information(kernel, grid, weights, X, alpha, delta) -> np.ndarray:
    """Empirical outer-product Fisher estimate, per chain.

    Named for what it is. Fisher's identity gives the exact per-chain score of the marginal
    log-likelihood from one BP pass, and ``sum_d g_d g_d^T / D`` estimates the information
    matrix -- but it is the outer-product form, not the negative Hessian, and it is an
    estimate rather than the observed information at the data. The distinction was blurred in
    the earlier write-up and is one of the things the review asked to be stated plainly.
    """
    grad = kernel.grad_log_transition_matrix(grid)
    log_k = kernel.log_transition_matrix(grid)
    g = np.stack([
        q_gradient(
            e_step_multi(grid, weights, log_k, [(x[None, :], alpha, delta)]),
            grad,
        )
        for x in X
    ])
    return g.T @ g / len(X)


def efficient_information(j_mat: np.ndarray) -> np.ndarray:
    """``I_{xx . eta} = 1 / (J^{-1})[x, x]`` for each coordinate.

    The information about one parameter that survives having to estimate the others. Equal to
    the diagonal only when the coordinates are orthogonal; strictly smaller otherwise, and the
    gap is precisely the nuisance coupling the diagonal comparison ignores.
    """
    inv = np.linalg.inv(j_mat)
    return 1.0 / np.diag(inv)


def part_information(cfg, out):
    grid, weights = make_grid(cfg["grid_a"], cfg["grid_size"])
    q_true = 1.0 - RHO_TRUE**2
    rows = []

    for beta_true in cfg["betas"]:
        kernel = GeneralizedGaussianKernel(RHO_TRUE, q_true, beta_true)
        kurt = generalized_gaussian_excess_kurtosis(beta_true)

        for t in cfg["t_values"]:
            alpha, delta = alpha_delta(t)
            rng = rng_for("exp22-info", beta_true, t)
            A = np.stack([
                sample_generalized_gaussian(rng, kernel, N_SITES)
                for _ in range(cfg["n_info"])
            ])
            X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

            j_mat = outer_product_information(kernel, grid, weights, X, alpha, delta)
            try:
                eff = efficient_information(j_mat)
                cond = float(np.linalg.cond(j_mat))
            except np.linalg.LinAlgError:
                eff = np.full(3, np.nan)
                cond = float("inf")

            # Correlation between coordinates, which is what makes the diagonal misleading.
            d = np.sqrt(np.diag(j_mat))
            corr = j_mat / np.outer(d, d)

            row = {
                "beta_true": beta_true,
                "excess_kurtosis": kurt,
                "t": t,
                "alpha": alpha,
                "delta": delta,
                "n_info": cfg["n_info"],
                "condition_number": cond,
                "corr_rho_beta": float(corr[0, 2]),
                "corr_q_beta": float(corr[1, 2]),
                "corr_rho_q": float(corr[0, 1]),
            }
            for i, name in enumerate(COORDS):
                row[f"diag_{name}"] = float(j_mat[i, i])
                row[f"eff_{name}"] = float(eff[i])
                # How much the diagonal overstates what is actually available.
                row[f"nuisance_cost_{name}"] = float(j_mat[i, i] / eff[i])
            rows.append(row)

            print(f"  beta={beta_true:4.2f} t={t:5.3f}  "
                  f"diag(rho,q,beta)=({j_mat[0,0]:.4g},{j_mat[1,1]:.4g},{j_mat[2,2]:.4g})  "
                  f"eff=({eff[0]:.4g},{eff[1]:.4g},{eff[2]:.4g})", flush=True)

    _plot(rows, cfg, out)
    return rows


def _plot(rows, cfg, out):
    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    beta0 = cfg["betas"][0]
    sub = sorted([r for r in rows if r["beta_true"] == beta0], key=lambda r: r["t"])
    ts = [r["t"] for r in sub]

    for name, style in zip(COORDS, ("o-", "s-", "^-")):
        ax[0].loglog(ts, [r[f"eff_{name}"] for r in sub], style, label=name, ms=4)
    ax[0].set_xlabel("noise level $t$")
    ax[0].set_ylabel("efficient information per chain")
    ax[0].set_title(rf"Nuisance-corrected information, $\beta={beta0}$")
    ax[0].legend()

    # Decay factor across the schedule: the number the claim is actually about.
    for name, style in zip(COORDS, ("o-", "s-", "^-")):
        decays = []
        for b in cfg["betas"]:
            s = sorted([r for r in rows if r["beta_true"] == b], key=lambda r: r["t"])
            if len(s) >= 2 and s[-1][f"eff_{name}"] > 0:
                decays.append(s[0][f"eff_{name}"] / s[-1][f"eff_{name}"])
            else:
                decays.append(np.nan)
        ax[1].semilogy(cfg["betas"], decays, style, label=name, ms=4)
    ax[1].set_xlabel(r"true shape $\beta$")
    ax[1].set_ylabel("information decay factor across $t$")
    ax[1].set_title("Which coordinate does the channel destroy fastest?")
    ax[1].legend()
    save_figure(fig, out / "shape_information.png")


def main() -> None:
    parser = experiment_parser(
        "exp_22_shape_information",
        "Efficient Fisher information for a genuine shape parameter at fixed variance.",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 201, "grid_a": 8.0, "n_info": 64,
        "betas": (1.2, 2.0), "t_values": (0.05, 0.4, 1.6),
    }
    full = {
        "grid_size": 401, "grid_a": 8.0, "n_info": 512,
        "betas": (1.0, 1.2, 1.5, 2.0, 3.0),
        "t_values": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "information": ("efficient information vs noise level",
                        lambda o: write_csv(o / "shape_information.csv",
                                            part_information(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "coords": list(COORDS),
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
