"""Experiment 04 -- Approximate Markovianity and informed score computation (Layer 4).

All priors here are Gaussian, so every score is available in closed form and
the "Markov approximation error" can be isolated exactly (no grid, no Monte
Carlo in the operator analysis).

Part 1 (Model A): AR(1) plus global latent,
    a = (y + beta g) / sqrt(1 + beta^2),   Cov = (rho^{|i-j|} + beta^2)/(1+beta^2).
  Scores compared:
    s_true    exact precision score;
    s_naive   chain BP ignoring the global component (prior Sigma_y/(1+beta^2));
    s_cl      chain BP under the best Markov (Chow-Liu) approximation, which
              matches all adjacent-pair covariances;
    s_aug     chain BP + closed-form rank-one Woodbury correction for the
              global latent == augmented BP with the latent integrated out;
              exact by construction (verified to machine precision).
  Analysis: relative residual norm ||s_true - s_markov|| / ||s_true||, the
  residual operator D_t = Sigma_markov,t^{-1} - Sigma_t^{-1}, its singular
  spectrum and effective rank, and the shape of its dominant singular vector.
  For the naive chain the residual is EXACTLY rank one (Woodbury); for
  Chow-Liu it is approximately rank one.

Part 2 (Model B): AR(1) precision plus weak long-range coupling,
    Q = Q_AR1 + gamma Q_long.
  Same analysis versus gamma: how fast does the residual stop being low-rank?

Part 3 (hybrid learning, Model A): supervised score regression with exact
  targets. Compared at equal data and equal parameters:
    (i)   direct MLP score  s_theta(x, t);
    (ii)  Markov BP baseline s_cl (no learning);
    (iii) hybrid s_cl + r_theta(x, t) with the SAME MLP architecture learning
          only the residual.
  Metrics: test relative score error vs training-set size (sample efficiency)
  and vs parameter count (parameter efficiency).
"""

from __future__ import annotations

import numpy as np

from common import experiment_parser, provenance
from src.exact_scores import precision_matrix_score, score_batch, sigma_t
from src.markov_approx import (
    chow_liu_chain_covariance,
    effective_rank,
    operator_spectrum,
    rank_one_global_correction,
    residual_operator,
)
from src.nnet import MLP, predict_score, train_score_net
from src.plotting import new_figure, save_figure
from src.priors import GaussianAR1, GaussianAR1PlusGlobal, GaussianLongRange
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 40
RHO = 0.8
T_VALUES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4)
BETAS = (0.1, 0.25, 0.5, 1.0)
GAMMAS = (0.05, 0.1, 0.2, 0.4)


# ----------------------------------------------------------------------------
# Parts 1 and 2: operator-level analysis (exact linear algebra)
# ----------------------------------------------------------------------------

def analyze_model(
    name: str,
    sigma_true: np.ndarray,
    sigma_markov: dict[str, np.ndarray],
    strength: float,
    t_values,
    n_mc: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Residual-operator and Monte Carlo residual-norm analysis for one model."""
    rows = []
    chol = np.linalg.cholesky(sigma_true)
    for t in t_values:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        # Monte Carlo x ~ p_t for realistic relative norms.
        A = chol @ rng.standard_normal((N_SITES, n_mc))
        X = (alpha * A + np.sqrt(delta) * rng.standard_normal((N_SITES, n_mc))).T
        S_true = score_batch(X, sigma_true, alpha, delta)

        for approx_name, sig_m in sigma_markov.items():
            D = residual_operator(sigma_true, sig_m, alpha, delta)
            sv = operator_spectrum(D)
            S_m = score_batch(X, sig_m, alpha, delta)
            res_norm = np.linalg.norm(S_true - S_m, axis=1)
            true_norm = np.linalg.norm(S_true, axis=1)
            rows.append({
                "model": name,
                "strength": strength,
                "approx": approx_name,
                "t": t,
                "rel_residual_median": float(np.median(res_norm / true_norm)),
                "sv1": float(sv[0]),
                "sv2": float(sv[1]),
                "sv2_over_sv1": float(sv[1] / sv[0]) if sv[0] > 0 else 0.0,
                "effective_rank": effective_rank(sv),
                "residual_fro": float(np.linalg.norm(sv)),
            })
    return rows


def part1_model_a(t_values, n_mc: int, out) -> list[dict]:
    rows: list[dict] = []
    check_rows: list[dict] = []
    for beta in BETAS:
        prior = GaussianAR1PlusGlobal(RHO, beta)
        sigma_true = prior.covariance(N_SITES)
        norm2 = 1.0 + beta**2
        sigma_naive = GaussianAR1(RHO).covariance(N_SITES) / norm2
        sigma_cl = chow_liu_chain_covariance(sigma_true)
        rng = rng_for("exp04-modelA", beta)
        rows += analyze_model(
            "A_beta", sigma_true,
            {"naive_chain": sigma_naive, "chow_liu": sigma_cl},
            beta, t_values, n_mc, rng,
        )
        # Augmented BP check: chain + rank-one Woodbury == exact.
        x = rng.standard_normal(N_SITES)
        for t in (t_values[0], t_values[-1]):
            alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
            s_true = precision_matrix_score(x, sigma_true, alpha, delta)
            s_chain = precision_matrix_score(x, sigma_naive, alpha, delta)
            corr = rank_one_global_correction(
                x, sigma_t(sigma_naive, alpha, delta),
                alpha**2 * beta**2 / norm2,
            )
            err = float(np.max(np.abs(s_chain + corr - s_true)))
            check_rows.append({"beta": beta, "t": t, "augmented_max_abs_err": err})
    write_csv(out / "modelA_augmented_bp_check.csv", check_rows)
    return rows


def part2_model_b(t_values, n_mc: int) -> list[dict]:
    rows: list[dict] = []
    for gamma in GAMMAS:
        prior = GaussianLongRange(RHO, gamma)
        sigma_true = prior.covariance(N_SITES)
        sigma_cl = chow_liu_chain_covariance(sigma_true)
        rng = rng_for("exp04-modelB", gamma)
        rows += analyze_model(
            "B_gamma", sigma_true, {"chow_liu": sigma_cl},
            gamma, t_values, n_mc, rng,
        )
    return rows


# ----------------------------------------------------------------------------
# Part 3: hybrid learning
# ----------------------------------------------------------------------------

def make_dataset(
    prior: GaussianAR1PlusGlobal,
    sigma_true: np.ndarray,
    sigma_markov: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    t_range=(0.02, 3.0),
):
    """(X, T, S_true, S_markov): observations, times, exact and Markov scores."""
    n = sigma_true.shape[0]
    chol = np.linalg.cholesky(sigma_true)
    T = np.exp(rng.uniform(np.log(t_range[0]), np.log(t_range[1]), size=n_samples))
    A = (chol @ rng.standard_normal((n, n_samples))).T
    alpha = np.exp(-T)[:, None]
    delta = (1.0 - np.exp(-2.0 * T))[:, None]
    X = alpha * A + np.sqrt(delta) * rng.standard_normal((n_samples, n))
    S_true = np.empty_like(X)
    S_markov = np.empty_like(X)
    for k in range(n_samples):
        a_k, d_k = float(alpha[k, 0]), float(delta[k, 0])
        S_true[k] = precision_matrix_score(X[k], sigma_true, a_k, d_k)
        S_markov[k] = precision_matrix_score(X[k], sigma_markov, a_k, d_k)
    return X, T, S_true, S_markov


def part3_hybrid_learning(out, quick: bool) -> None:
    beta = 0.5
    prior = GaussianAR1PlusGlobal(RHO, beta)
    sigma_true = prior.covariance(N_SITES)
    sigma_cl = chow_liu_chain_covariance(sigma_true)

    train_sizes = (128, 1024) if quick else (64, 256, 1024, 4096)
    widths = (64,) if quick else (16, 64, 256)
    n_test = 512 if quick else 2048
    n_steps = 800 if quick else 4000

    rng_data = rng_for("exp04-learn-data")
    X_te, T_te, S_te, S_m_te = make_dataset(
        prior, sigma_true, sigma_cl, n_test, rng_data
    )
    baseline_rel = float(
        np.linalg.norm(S_m_te - S_te) / np.linalg.norm(S_te)
    )

    rows = []
    # Sample efficiency at fixed width 64.
    for n_train in train_sizes:
        X_tr, T_tr, S_tr, S_m_tr = make_dataset(
            prior, sigma_true, sigma_cl, n_train, rng_for("exp04-train", n_train)
        )
        for mode in ("direct", "hybrid"):
            net = MLP.init((N_SITES + 3, 64, 64, N_SITES), rng_for("init", mode, n_train))
            target = S_tr if mode == "direct" else S_tr - S_m_tr
            net = train_score_net(
                net, X_tr, T_tr, target, rng_for("batch", mode, n_train),
                n_steps=n_steps,
            )
            pred = predict_score(net, X_te, T_te)
            if mode == "hybrid":
                pred = pred + S_m_te
            rel = float(np.linalg.norm(pred - S_te) / np.linalg.norm(S_te))
            rows.append({
                "sweep": "sample_efficiency", "mode": mode, "n_train": n_train,
                "width": 64, "n_params": net.n_params, "test_rel_error": rel,
                "markov_baseline_rel_error": baseline_rel,
            })

    # Parameter efficiency at fixed n_train.
    n_fixed = 1024 if 1024 in train_sizes else train_sizes[-1]
    X_tr, T_tr, S_tr, S_m_tr = make_dataset(
        prior, sigma_true, sigma_cl, n_fixed, rng_for("exp04-train", n_fixed)
    )
    for width in widths:
        for mode in ("direct", "hybrid"):
            net = MLP.init(
                (N_SITES + 3, width, width, N_SITES), rng_for("init-w", mode, width)
            )
            target = S_tr if mode == "direct" else S_tr - S_m_tr
            net = train_score_net(
                net, X_tr, T_tr, target, rng_for("batch-w", mode, width),
                n_steps=n_steps,
            )
            pred = predict_score(net, X_te, T_te)
            if mode == "hybrid":
                pred = pred + S_m_te
            rel = float(np.linalg.norm(pred - S_te) / np.linalg.norm(S_te))
            rows.append({
                "sweep": "parameter_efficiency", "mode": mode, "n_train": n_fixed,
                "width": width, "n_params": net.n_params, "test_rel_error": rel,
                "markov_baseline_rel_error": baseline_rel,
            })

    write_csv(out / "hybrid_learning.csv", rows)

    # Figures.
    fig, ax = new_figure()
    for mode, marker in [("direct", "o"), ("hybrid", "s")]:
        sub = [r for r in rows if r["sweep"] == "sample_efficiency" and r["mode"] == mode]
        ax.plot([r["n_train"] for r in sub], [r["test_rel_error"] for r in sub],
                marker=marker, label=f"{mode} MLP")
    ax.axhline(baseline_rel, color="k", linestyle=":",
               label="Markov BP baseline (no learning)")
    ax.set(xscale="log", yscale="log", xlabel="training set size",
           ylabel="test relative score error",
           title=f"Sample efficiency (Model A, beta = {beta}, rho = {RHO})")
    ax.legend()
    save_figure(fig, out / "hybrid_sample_efficiency.png")

    fig, ax = new_figure()
    for mode, marker in [("direct", "o"), ("hybrid", "s")]:
        sub = [r for r in rows if r["sweep"] == "parameter_efficiency" and r["mode"] == mode]
        ax.plot([r["n_params"] for r in sub], [r["test_rel_error"] for r in sub],
                marker=marker, label=f"{mode} MLP")
    ax.axhline(baseline_rel, color="k", linestyle=":",
               label="Markov BP baseline (no learning)")
    ax.set(xscale="log", yscale="log", xlabel="number of MLP parameters",
           ylabel="test relative score error",
           title=f"Parameter efficiency (n_train = {n_fixed})")
    ax.legend()
    save_figure(fig, out / "hybrid_parameter_efficiency.png")


# ----------------------------------------------------------------------------
# Figures for parts 1-2
# ----------------------------------------------------------------------------

def make_operator_figures(rows, out) -> None:
    # Residual norm vs t for Model A (Chow-Liu), by beta.
    fig, ax = new_figure()
    for beta in BETAS:
        sub = [r for r in rows if r["model"] == "A_beta" and r["strength"] == beta
               and r["approx"] == "chow_liu"]
        ax.plot([r["t"] for r in sub], [r["rel_residual_median"] for r in sub],
                marker="o", label=f"beta = {beta}")
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="median ||s_true - s_markov|| / ||s_true||",
           title="Model A: best-chain (Chow-Liu) score residual")
    ax.legend()
    save_figure(fig, out / "modelA_residual_norm.png")

    # Rank-one dominance: sv2/sv1 vs t.
    fig, ax = new_figure()
    for beta in BETAS:
        for approx, ls in [("naive_chain", "--"), ("chow_liu", "-")]:
            sub = [r for r in rows if r["model"] == "A_beta"
                   and r["strength"] == beta and r["approx"] == approx]
            ax.plot([r["t"] for r in sub], [r["sv2_over_sv1"] for r in sub],
                    linestyle=ls, marker=".",
                    label=f"beta={beta}, {approx}" if beta in (0.25, 1.0) else None)
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="sigma_2 / sigma_1 of residual operator",
           title="Model A: the non-Markov score correction is (near) rank one")
    ax.legend(fontsize=8)
    save_figure(fig, out / "modelA_rank_one_dominance.png")

    # Model B: effective rank vs gamma.
    fig, ax = new_figure()
    for gamma in GAMMAS:
        sub = [r for r in rows if r["model"] == "B_gamma" and r["strength"] == gamma]
        ax.plot([r["t"] for r in sub], [r["effective_rank"] for r in sub],
                marker="o", label=f"gamma = {gamma}")
    ax.set(xscale="log", xlabel="diffusion time t",
           ylabel="effective rank of residual operator",
           title="Model B: complexity of the long-range score correction")
    ax.legend()
    save_figure(fig, out / "modelB_effective_rank.png")

    fig, ax = new_figure()
    for gamma in GAMMAS:
        sub = [r for r in rows if r["model"] == "B_gamma" and r["strength"] == gamma]
        ax.plot([r["t"] for r in sub], [r["rel_residual_median"] for r in sub],
                marker="o", label=f"gamma = {gamma}")
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="median relative residual",
           title="Model B: Chow-Liu chain score residual")
    ax.legend()
    save_figure(fig, out / "modelB_residual_norm.png")


def main() -> None:
    args = experiment_parser("exp_04_approx_markovianity", __doc__).parse_args()
    out = ensure_dir(args.output_dir)
    t_values = T_VALUES[::2] if args.quick else T_VALUES
    n_mc = 64 if args.quick else 256

    write_json(out / "params.json", {
        "n_sites": N_SITES, "rho": RHO, "betas": BETAS, "gammas": GAMMAS,
        "t_values": t_values, "n_mc": n_mc, "provenance": provenance(),
    })

    rows = part1_model_a(t_values, n_mc, out)
    rows += part2_model_b(t_values, n_mc)
    write_csv(out / "residual_operator_analysis.csv", rows)
    make_operator_figures(rows, out)

    part3_hybrid_learning(out, args.quick)
    print(f"exp_04 done -> {out}")


if __name__ == "__main__":
    main()
