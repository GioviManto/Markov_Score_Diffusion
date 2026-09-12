"""Experiment 21 -- What happens when the data is not Markov?

The largest gap in the story. Every result so far was measured on data the estimator can
represent exactly: a chain fitted to a chain. That makes the structured estimator's advantage
over the networks a statement about sample efficiency under a *correct* prior, and says
nothing about what happens when the prior is wrong -- which is the situation any real
application is in. `exp_04` analyses the score residual of known non-Markov Gaussian priors,
but nothing has ever fitted a kernel to data outside the chain family.

The comparison is deliberately set up so it can go against us. The estimator is handed a
structural assumption that is now false, while the CNN and MLP are handed none, so as the
violation grows the advantage must shrink and at some point invert. Finding where it inverts
is the result; asserting that it does not would be the failure.

Parts
-----
gauss      Gaussian non-Markov priors, where the reference is exact linear algebra. Two
           mechanisms, swept independently: a shared global latent (`GaussianAR1PlusGlobal`,
           parameter beta) and weak long-range precision coupling (`GaussianLongRange`,
           parameter gamma). beta = gamma = 0 recovers the Markov chain and must reproduce
           the existing numbers -- the control that says the harness is sound.

           Also asks what EM *converges to*, which has a falsifiable answer. Maximum
           marginal likelihood inside the chain family should land on the best-Markov
           projection of the true prior, and for Gaussian data that projection is available
           in closed form from `markov_approx.chow_liu_chain_covariance`. Comparing the
           fitted rho against the Chow-Liu rho is a prediction with a way to be wrong.

laplace    The corner that matters and the one nothing else covers: non-Gaussian *and*
           non-Markov. A Gaussian non-Markov prior is a poor test of a method whose whole
           point is representing non-Gaussian innovations, because a Gaussian baseline is
           already exact there. `src/nonmarkov.py` supplies an exact reference for a Laplace
           chain plus a global latent by conditioning on the latent and marginalising it
           with Gauss-Hermite, so this part is measured against truth, not against a proxy.

Both parts use the validation protocol of Track 1: the networks' parameterisation is chosen
on a held-out validation bundle, never on the test bundle they are then scored on.
"""

from __future__ import annotations

import time

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import grid_bp_batch, make_grid
from src.denoiser import dsm_posterior_mean, train_dsm_denoiser
from src.em import fit_em
from src.exact_scores import exact_gaussian_posterior_mean
from src.kernels import MixtureInnovationKernel
from src.local_head import local_posterior_mean, train_local_head
from src.markov_approx import chow_liu_chain_covariance
from src.noising import alpha_delta
from src.nonmarkov import global_latent_posterior_mean, laplace_plus_global
from src.plotting import new_figure, save_figure
from src.priors import GaussianAR1PlusGlobal, GaussianLongRange
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 32
RHO = 0.85
T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)
GRID_A = 8.0


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _noised_groups(A, t_values, rng):
    groups = []
    for t in t_values:
        alpha, delta = alpha_delta(t)
        groups.append((alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape),
                       alpha, delta))
    return groups


def _fit_all_arms(A, grid, weights, cfg, tag):
    """EM-BP plus both network families, on identical chains.

    Identical data is the point: the estimator's prior is wrong here and the networks' is
    absent, so any difference is about what each does with the same information.
    """
    rng = rng_for("exp21-noise", tag)
    t0 = time.perf_counter()
    kernel, trace = fit_em(
        MixtureInnovationKernel.init(
            cfg["em_components"], rho=0.3, var=0.8, rng=rng_for("exp21-eminit", tag)
        ),
        grid, weights, _noised_groups(A, T_TRAIN, rng), n_iters=cfg["em_iters"],
    )
    em_seconds = time.perf_counter() - t0

    # Continuous-time training, matching the P0.B fix in exp_16. This experiment exists to
    # ask whether the structural prior still pays when the data is not Markov; training the
    # networks on five discrete levels would hand them an unrelated handicap and make the
    # crossover it measures meaningless.
    t_range = (cfg["t_min"], cfg["t_max"])
    nets = {}
    for mode in ("eps", "x0"):
        nets[("mlp", mode)] = train_dsm_denoiser(
            A, T_TRAIN, rng_for("exp21-mlp", tag, mode),
            hidden=cfg["net_hidden"], n_steps=cfg["net_steps"], parameterization=mode,
            t_range=t_range,
        )
        nets[("cnn", mode)] = train_local_head(
            A, T_TRAIN, cfg["cnn_radius"], rng_for("exp21-cnn", tag, mode),
            hidden=cfg["cnn_hidden"], n_steps=cfg["net_steps"], parameterization=mode,
            t_range=t_range,
        )
    return kernel, nets, trace, em_seconds


def _select_modes(nets, X_val, m_val, t):
    """Choose eps or x0 per family on the VALIDATION bundle -- never on test.

    Track 1's protocol, imported here rather than reinvented: the networks are the arms that
    stand to gain from an oracle, so leaving one in place would understate exactly the effect
    this experiment is looking for.
    """
    best = {}
    for family, predict in (("mlp", dsm_posterior_mean), ("cnn", local_posterior_mean)):
        errs = {
            mode: float(np.mean((predict(nets[(family, mode)], X_val, t) - m_val) ** 2))
            for mode in ("eps", "x0")
        }
        best[family] = min(errs, key=errs.get)
    return best


def _rel_error(m_hat, m_ref):
    return float(np.linalg.norm(m_hat - m_ref) / np.linalg.norm(m_ref))


def _fitted_rho_of_chow_liu(sigma0):
    """The lag-1 correlation of the best-Markov approximation to a Gaussian prior.

    Chow-Liu on a chain matches every adjacent-pair covariance exactly, so with a unit
    diagonal the implied autoregressive coefficient is simply that lag-1 entry. This is what
    maximum marginal likelihood inside the chain family should converge to.
    """
    cl = chow_liu_chain_covariance(sigma0)
    d = np.sqrt(np.diag(cl))
    corr = cl / np.outer(d, d)
    return float(np.mean(np.diag(corr, 1)))


# ---------------------------------------------------------------------------
# Part 1: Gaussian non-Markov, exact reference
# ---------------------------------------------------------------------------

def part1_gauss(cfg, out):
    """Sweep the violation and watch the advantage close, against an exact reference."""
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []

    families = [("global", b, GaussianAR1PlusGlobal(RHO, b)) for b in cfg["betas"]]
    families += [("longrange", g, GaussianLongRange(RHO, g)) for g in cfg["gammas"]]

    for mech, strength, prior in families:
        sigma0 = prior.covariance(N_SITES)
        tag = f"{mech}{strength:g}"

        # Seeded on the MECHANISM only, deliberately not on the strength. Both priors draw
        # their randomness the same way at every strength -- GaussianAR1PlusGlobal draws the
        # y-chain then the latent g, GaussianLongRange draws standard normals then applies a
        # gamma-dependent Cholesky factor -- so a strength-independent seed gives common
        # random numbers: identical underlying draws, only the mixing changes. That makes the
        # sweep a within-data contrast.
        #
        # The committed gauss run (SLURM 618380) predates this and used a strength-dependent
        # seed, which is why its beta=0.1 cell shows a lower EM error than the beta=0 control.
        # That is a data draw, not a finding, and the trend across the sweep is far larger
        # than the scatter it introduces.
        rng = rng_for("exp21-data", mech)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
        A_val = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_val"])])
        A_test = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_test"])])

        kernel, nets, trace, em_seconds = _fit_all_arms(A, grid, weights, cfg, tag)
        cl_rho = _fitted_rho_of_chow_liu(sigma0)

        for t in T_TRAIN:
            alpha, delta = alpha_delta(t)
            rng_e = rng_for("exp21-eval", tag, t)
            X_val = alpha * A_val + np.sqrt(delta) * rng_e.standard_normal(A_val.shape)
            X_test = alpha * A_test + np.sqrt(delta) * rng_e.standard_normal(A_test.shape)

            # The reference is exact: Gaussian prior, so one linear solve per chain.
            m_val = np.stack(
                [exact_gaussian_posterior_mean(x, sigma0, alpha, delta) for x in X_val])
            m_test = np.stack(
                [exact_gaussian_posterior_mean(x, sigma0, alpha, delta) for x in X_test])

            best = _select_modes(nets, X_val, m_val, t)
            m_em = grid_bp_batch(grid, weights, kernel.log_transition_matrix(grid),
                                 X_test, alpha, delta)[0]

            errs = {
                "em_bp": _rel_error(m_em, m_test),
                "cnn": _rel_error(
                    local_posterior_mean(nets[("cnn", best["cnn"])], X_test, t), m_test),
                "mlp": _rel_error(
                    dsm_posterior_mean(nets[("mlp", best["mlp"])], X_test, t), m_test),
            }
            for arm, err in errs.items():
                rows.append({
                    "mechanism": mech, "strength": strength, "arm": arm, "t": t,
                    "score_rel_l2": err,
                    "ratio_to_em": err / errs["em_bp"],
                    "fitted_rho": float(kernel.rho),
                    "chow_liu_rho": cl_rho,
                    "rho_minus_chow_liu": float(kernel.rho) - cl_rho,
                    "em_seconds": em_seconds,
                    "em_iters": len(trace.log_evidence),
                    "em_monotone_violation": trace.monotone_violation,
                    "mode_cnn": best["cnn"], "mode_mlp": best["mlp"],
                    **kernel.innovation_moments,
                })
            print(f"  {mech:10s} s={strength:5.2f} t={t:4.2f}  "
                  f"em={errs['em_bp']:.5f} cnn={errs['cnn']:.5f} mlp={errs['mlp']:.5f}  "
                  f"cnn/em={errs['cnn']/errs['em_bp']:6.2f}  "
                  f"rho={float(kernel.rho):.4f} vs CL {cl_rho:.4f}", flush=True)

    _plot_gauss(rows, cfg, out)
    return rows


def _plot_gauss(rows, cfg, out):
    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    for mech, strengths in (("global", cfg["betas"]), ("longrange", cfg["gammas"])):
        for arm, style in (("cnn", "s-"), ("mlp", "^-")):
            ratios = []
            for s in strengths:
                vals = [r["ratio_to_em"] for r in rows
                        if r["mechanism"] == mech and r["strength"] == s
                        and r["arm"] == arm]
                ratios.append(float(np.mean(vals)) if vals else np.nan)
            ax[0].plot(strengths, ratios, style, label=f"{mech} / {arm}", ms=4)
    ax[0].axhline(1.0, color="k", lw=1, ls=":")
    ax[0].set_xlabel("violation strength ($\\beta$ or $\\gamma$)")
    ax[0].set_ylabel("network error / EM-BP error")
    ax[0].set_title("Where the structural prior stops paying")
    ax[0].legend(fontsize=7)

    for mech, strengths in (("global", cfg["betas"]), ("longrange", cfg["gammas"])):
        fitted, chow = [], []
        for s in strengths:
            vals = [r for r in rows if r["mechanism"] == mech and r["strength"] == s]
            fitted.append(vals[0]["fitted_rho"] if vals else np.nan)
            chow.append(vals[0]["chow_liu_rho"] if vals else np.nan)
        ax[1].plot(strengths, fitted, "o-", ms=4, label=f"{mech}: EM")
        ax[1].plot(strengths, chow, "k--", lw=1, label=f"{mech}: Chow-Liu")
    ax[1].set_xlabel("violation strength")
    ax[1].set_ylabel(r"lag-1 coefficient $\rho$")
    ax[1].set_title("Does EM find the best-Markov projection?")
    ax[1].legend(fontsize=7)
    save_figure(fig, out / "nonmarkov_gauss.png")


# ---------------------------------------------------------------------------
# Part 2: non-Gaussian and non-Markov
# ---------------------------------------------------------------------------

def part2_laplace(cfg, out):
    """The corner nothing else covers, against the exact g-marginalised reference."""
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []

    for beta in cfg["betas"]:
        prior = laplace_plus_global(RHO, beta)
        tag = f"laplace_global{beta:g}"

        # Strength-independent, for the same common-random-numbers reason as part 1.
        rng = rng_for("exp21-lap-data")
        A = prior.sample_batch(rng, cfg["n_chains"], N_SITES)
        A_val = prior.sample_batch(rng, cfg["n_val"], N_SITES)
        A_test = prior.sample_batch(rng, cfg["n_test"], N_SITES)

        kernel, nets, trace, em_seconds = _fit_all_arms(A, grid, weights, cfg, tag)

        for t in T_TRAIN:
            alpha, delta = alpha_delta(t)
            rng_e = rng_for("exp21-lap-eval", tag, t)
            X_val = alpha * A_val + np.sqrt(delta) * rng_e.standard_normal(A_val.shape)
            X_test = alpha * A_test + np.sqrt(delta) * rng_e.standard_normal(A_test.shape)

            m_val, _ = global_latent_posterior_mean(
                prior, grid, weights, X_val, alpha, delta, cfg["n_nodes"])
            m_test, _ = global_latent_posterior_mean(
                prior, grid, weights, X_test, alpha, delta, cfg["n_nodes"])

            best = _select_modes(nets, X_val, m_val, t)
            m_em = grid_bp_batch(grid, weights, kernel.log_transition_matrix(grid),
                                 X_test, alpha, delta)[0]

            errs = {
                "em_bp": _rel_error(m_em, m_test),
                "cnn": _rel_error(
                    local_posterior_mean(nets[("cnn", best["cnn"])], X_test, t), m_test),
                "mlp": _rel_error(
                    dsm_posterior_mean(nets[("mlp", best["mlp"])], X_test, t), m_test),
            }
            for arm, err in errs.items():
                rows.append({
                    "beta": beta, "arm": arm, "t": t,
                    "score_rel_l2": err,
                    "ratio_to_em": err / errs["em_bp"],
                    "fitted_rho": float(kernel.rho),
                    "em_seconds": em_seconds,
                    "em_iters": len(trace.log_evidence),
                    "em_monotone_violation": trace.monotone_violation,
                    "n_nodes": cfg["n_nodes"],
                    "mode_cnn": best["cnn"], "mode_mlp": best["mlp"],
                    **kernel.innovation_moments,
                })
            print(f"  beta={beta:5.2f} t={t:4.2f}  em={errs['em_bp']:.5f} "
                  f"cnn={errs['cnn']:.5f} mlp={errs['mlp']:.5f}  "
                  f"cnn/em={errs['cnn']/errs['em_bp']:6.2f}", flush=True)

    fig, ax = new_figure()
    for arm, style in (("cnn", "s-"), ("mlp", "^-")):
        ratios = [float(np.mean([r["ratio_to_em"] for r in rows
                                 if r["beta"] == b and r["arm"] == arm]))
                  for b in cfg["betas"]]
        ax.plot(cfg["betas"], ratios, style, label=arm, ms=4)
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xlabel(r"global-latent strength $\beta$")
    ax.set_ylabel("network error / EM-BP error")
    ax.set_title("Laplace chain plus a global latent, against the exact reference")
    ax.legend()
    save_figure(fig, out / "nonmarkov_laplace.png")
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_21_nonmarkov",
        "What does the chain estimator do when the data is not a chain?",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 201, "n_chains": 128, "n_val": 64, "n_test": 64,
        "betas": (0.0, 0.5), "gammas": (0.0, 0.2),
        "em_components": 4, "em_iters": 10,
        "net_hidden": (64, 64), "cnn_hidden": (64, 64), "net_steps": 500,
        "cnn_radius": 6, "n_nodes": 21, "t_min": 0.02, "t_max": 3.0,
    }
    full = {
        "grid_size": 401, "n_chains": 2048, "n_val": 256, "n_test": 256,
        "betas": (0.0, 0.1, 0.25, 0.5, 1.0), "gammas": (0.0, 0.05, 0.1, 0.2, 0.4),
        "em_components": 8, "em_iters": 80,
        "net_hidden": (128, 128), "cnn_hidden": (64, 64), "net_steps": 8000,
        "cnn_radius": 6, "n_nodes": 41, "t_min": 0.02, "t_max": 3.0,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "gauss": ("Gaussian non-Markov, exact linear-algebra reference",
                  lambda o: write_csv(o / "nonmarkov_gauss.csv", part1_gauss(cfg, o))),
        "laplace": ("non-Gaussian and non-Markov, exact marginalised reference",
                    lambda o: write_csv(o / "nonmarkov_laplace.csv",
                                        part2_laplace(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "rho": RHO, "t_train": T_TRAIN, "grid_half_width": GRID_A,
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
