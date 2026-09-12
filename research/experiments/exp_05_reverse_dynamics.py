"""Experiment 05 -- Dynamics of reverse diffusion under approximate scores.

Convention: forward dX = -X dt + sqrt(2) dW (see src/reverse.py). All samplers
integrate from t = T down to t = t_min on a geometric time grid. Paired
comparisons use common random numbers (identical Brownian increments and
identical initial noise across score functions).

Setup 1 (Laplace chain, rho = 0.85): generation from x_T ~ N(0, I).
  Scores: grid BP (reference) vs analytic Gaussian BP (== matched Gaussian
  model). Questions:
    - Do reverse samples recover the heavy-tailed innovations (excess kurtosis
      of a_i - rho a_{i-1}) or does the Gaussian score wash them out?
    - Is the small pointwise score difference at moderate t dynamically stable,
      or does it accumulate along the trajectory?
  Tracked along reverse time: trajectory divergence between the two samplers,
  pointwise score deviation evaluated ON the reference trajectory, score norm,
  mean BP posterior variance (from grid BP beliefs).

Setup 2 (Laplace chain): reconstruction. Start from a forward-noised known
  clean sample at t = T_rec and denoise via the probability-flow ODE; track
  correlation and MSE with the clean sample over reverse time.

Setup 3 (Model A, Gaussian AR(1) + global latent, beta = 0.5): generation with
  s_true vs best-chain Markov score vs hybrid (chain + rank-one Woodbury ==
  exact augmented BP). Because everything is Gaussian, terminal sample
  statistics have exact targets: we report the top eigenvalue of the sample
  covariance (the global mode that a Markov score cannot produce) and the
  Frobenius error to Sigma_0.
"""

from __future__ import annotations

import numpy as np

from common import experiment_parser, provenance
from src.bp_gaussian import gaussian_chain_bp
from src.bp_grid import grid_bp_batch, make_grid
from src.exact_scores import score_batch, sigma_t
from src.markov_approx import chow_liu_chain_covariance, rank_one_global_correction
from src.plotting import new_figure, save_figure
from src.priors import GaussianAR1, GaussianAR1PlusGlobal, LaplaceAR1
from src.reverse import time_grid
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 32
RHO = 0.85
T_MAX, T_MIN = 3.0, 0.01


# ----------------------------------------------------------------------------
# Score functions (batched: X of shape (B, n) -> scores (B, n))
# ----------------------------------------------------------------------------

def laplace_grid_score_fn(prior: LaplaceAR1, m_grid: int = 401, a_width: float = 8.0):
    grid, w = make_grid(a_width, m_grid)
    log_K = prior.log_transition_matrix(grid)

    def fn(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        means, variances = grid_bp_batch(grid, w, log_K, X, alpha, delta)
        return -(X - alpha * means) / delta, variances

    return fn


def laplace_gauss_score_fn(prior: LaplaceAR1):
    def fn(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        means = np.empty_like(X)
        variances = np.empty_like(X)
        for b in range(X.shape[0]):
            res = gaussian_chain_bp(X[b], prior.rho, prior.q, alpha, delta)
            means[b] = res.means
            variances[b] = res.variances
        return -(X - alpha * means) / delta, variances

    return fn


def gaussian_matrix_score_fn(sigma0: np.ndarray):
    def fn(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        return score_batch(X, sigma0, alpha, delta), np.full_like(X, np.nan)

    return fn


def hybrid_chain_plus_rank_one_fn(sigma_chain: np.ndarray, coupling: float):
    """Chain score + closed-form Woodbury rank-one global correction."""

    def fn(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        s_chain = score_batch(X, sigma_chain, alpha, delta)
        st = sigma_t(sigma_chain, alpha, delta)
        corr = np.stack([
            rank_one_global_correction(x, st, alpha**2 * coupling) for x in X
        ])
        return s_chain + corr, np.full_like(X, np.nan)

    return fn


# ----------------------------------------------------------------------------
# Reverse SDE loop with tracking (batched, common noise)
# ----------------------------------------------------------------------------

def reverse_sde_tracked(
    x_init: np.ndarray,
    score_fn,
    times: np.ndarray,
    rng: np.random.Generator,
    a_clean: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Euler-Maruyama reverse SDE for a batch, recording summary stats per step."""
    x = x_init.copy()
    records: list[dict] = []
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s, post_var = score_fn(x, float(t_now))
        has_var = bool(np.any(np.isfinite(post_var)))
        rec = {
            "t": float(t_now),
            "score_norm_mean": float(np.mean(np.linalg.norm(s, axis=1))),
            "posterior_var_mean": float(np.nanmean(post_var)) if has_var else np.nan,
            "x_var": float(np.var(x)),
        }
        if a_clean is not None:
            num = np.sum((x - x.mean(1, keepdims=True))
                         * (a_clean - a_clean.mean(1, keepdims=True)), axis=1)
            den = (np.linalg.norm(x - x.mean(1, keepdims=True), axis=1)
                   * np.linalg.norm(a_clean - a_clean.mean(1, keepdims=True), axis=1))
            rec["corr_with_clean_mean"] = float(np.mean(num / np.maximum(den, 1e-12)))
            rec["mse_vs_clean"] = float(np.mean((x - a_clean) ** 2))
        records.append(rec)
        noise = rng.standard_normal(x.shape)
        x = x + h * (x + 2.0 * s) + np.sqrt(2.0 * h) * noise
    return x, records


def innovation_excess_kurtosis(samples: np.ndarray, rho: float) -> float:
    eps = samples[:, 1:] - rho * samples[:, :-1]
    e = eps.ravel()
    e = e - e.mean()
    return float(np.mean(e**4) / np.mean(e**2) ** 2 - 3.0)


# ----------------------------------------------------------------------------
# Setups
# ----------------------------------------------------------------------------

def setup1_laplace_generation(out, n_batch: int, n_steps: int) -> None:
    prior = LaplaceAR1(RHO)
    times = time_grid(T_MAX, T_MIN, n_steps)
    fn_grid = laplace_grid_score_fn(prior)
    fn_gauss = laplace_gauss_score_fn(prior)

    x_init = rng_for("exp05-init").standard_normal((n_batch, N_SITES))
    x_ref, rec_ref = reverse_sde_tracked(
        x_init, fn_grid, times, rng_for("exp05-noise")
    )
    x_gauss, rec_gauss = reverse_sde_tracked(
        x_init, fn_gauss, times, rng_for("exp05-noise")  # same noise stream
    )

    # Pointwise score deviation ON the reference trajectory vs cumulative
    # trajectory divergence (the dynamical-stability question).
    div_rows = []
    x = x_init.copy()
    rng_n = rng_for("exp05-noise")
    x_alt = x_init.copy()
    rng_alt = rng_for("exp05-noise")
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s_ref, _ = fn_grid(x, float(t_now))
        s_g_on_ref, _ = fn_gauss(x, float(t_now))
        s_g_alt, _ = fn_gauss(x_alt, float(t_now))
        div_rows.append({
            "t": float(t_now),
            "pointwise_rel_score_dev": float(
                np.linalg.norm(s_g_on_ref - s_ref) / np.linalg.norm(s_ref)
            ),
            "trajectory_rel_divergence": float(
                np.linalg.norm(x_alt - x) / np.linalg.norm(x)
            ),
        })
        noise = rng_n.standard_normal(x.shape)
        x = x + h * (x + 2.0 * s_ref) + np.sqrt(2.0 * h) * noise
        noise_alt = rng_alt.standard_normal(x.shape)
        x_alt = x_alt + h * (x_alt + 2.0 * s_g_alt) + np.sqrt(2.0 * h) * noise_alt
    write_csv(out / "setup1_divergence.csv", div_rows)

    # Terminal statistics.
    clean = np.stack([prior.sample(rng_for("exp05-clean", k), N_SITES)
                      for k in range(n_batch)])
    stats = [{
        "sampler": name,
        "innovation_excess_kurtosis": innovation_excess_kurtosis(xs, RHO),
        "marginal_excess_kurtosis": float(
            np.mean(xs.ravel()**4) / np.mean(xs.ravel()**2) ** 2 - 3.0
        ),
        "marginal_var": float(np.var(xs)),
        "lag1_corr": float(np.corrcoef(xs[:, :-1].ravel(), xs[:, 1:].ravel())[0, 1]),
    } for name, xs in [
        ("grid_bp_score", x_ref),
        ("gaussian_bp_score", x_gauss),
        ("clean_prior_samples", clean),
    ]]
    write_csv(out / "setup1_terminal_stats.csv", stats)

    for rec, name in [(rec_ref, "grid"), (rec_gauss, "gauss")]:
        for r in rec:
            r["sampler"] = name
    write_csv(out / "setup1_tracking.csv", rec_ref + rec_gauss)

    # Figures.
    fig, ax = new_figure()
    ts = [r["t"] for r in div_rows]
    ax.plot(ts, [r["pointwise_rel_score_dev"] for r in div_rows], marker=".",
            label="pointwise score deviation (on reference path)")
    ax.plot(ts, [r["trajectory_rel_divergence"] for r in div_rows], marker=".",
            label="cumulative trajectory divergence (common noise)")
    ax.set(xscale="log", yscale="log", xlabel="reverse time t (integrating right to left)",
           ylabel="relative deviation",
           title="Laplace chain: pointwise score error vs dynamical divergence")
    ax.legend()
    save_figure(fig, out / "setup1_pointwise_vs_dynamic.png")

    fig, ax = new_figure()
    labels = [s["sampler"] for s in stats]
    vals = [s["innovation_excess_kurtosis"] for s in stats]
    ax.bar(labels, vals, color=["C0", "C1", "C2"])
    ax.axhline(3.0, color="k", linestyle=":", label="Laplace innovation kurtosis (+3)")
    ax.set(ylabel="excess kurtosis of fitted innovations",
           title="Do reverse samples recover heavy-tailed innovations?")
    ax.legend()
    save_figure(fig, out / "setup1_innovation_kurtosis.png")


def setup2_laplace_reconstruction(out, n_batch: int, n_steps: int) -> None:
    prior = LaplaceAR1(RHO)
    t_rec = 1.0
    times = time_grid(t_rec, T_MIN, n_steps)
    fn_grid = laplace_grid_score_fn(prior)
    fn_gauss = laplace_gauss_score_fn(prior)

    a_clean = np.stack([prior.sample(rng_for("exp05-rec-clean", k), N_SITES)
                        for k in range(n_batch)])
    alpha, delta = float(np.exp(-t_rec)), float(1.0 - np.exp(-2.0 * t_rec))
    x_start = (alpha * a_clean + np.sqrt(delta)
               * rng_for("exp05-rec-noise").standard_normal(a_clean.shape))

    rows = []
    for name, fn in [("grid", fn_grid), ("gauss", fn_gauss)]:
        _, rec = reverse_sde_tracked(
            x_start, fn, times, rng_for("exp05-rec-sde"), a_clean=a_clean
        )
        for r in rec:
            r["sampler"] = name
        rows += rec
    write_csv(out / "setup2_reconstruction.csv", rows)

    fig, ax = new_figure()
    for name, ls in [("grid", "-"), ("gauss", "--")]:
        sub = [r for r in rows if r["sampler"] == name]
        ax.plot([r["t"] for r in sub], [r["corr_with_clean_mean"] for r in sub],
                linestyle=ls, marker=".", label=f"{name} BP score")
    ax.set(xscale="log", xlabel="reverse time t", ylabel="corr(x_t, clean a)",
           title=f"Reconstruction from t = {t_rec}: correlation recovery")
    ax.invert_xaxis()
    ax.legend()
    save_figure(fig, out / "setup2_correlation_recovery.png")


def setup3_model_a(out, n_batch: int, n_steps: int) -> None:
    beta = 0.5
    prior = GaussianAR1PlusGlobal(0.8, beta)
    sigma_true = prior.covariance(N_SITES)
    sigma_cl = chow_liu_chain_covariance(sigma_true)
    norm2 = 1.0 + beta**2
    sigma_naive = GaussianAR1(0.8).covariance(N_SITES) / norm2

    fns = {
        "exact": gaussian_matrix_score_fn(sigma_true),
        "markov_chow_liu": gaussian_matrix_score_fn(sigma_cl),
        "markov_naive": gaussian_matrix_score_fn(sigma_naive),
        "hybrid_rank_one": hybrid_chain_plus_rank_one_fn(
            sigma_naive, beta**2 / norm2
        ),
    }
    times = time_grid(T_MAX, T_MIN, n_steps)
    x_init = rng_for("exp05-A-init").standard_normal((n_batch, N_SITES))

    rows = []
    eig_true = float(np.linalg.eigvalsh(sigma_true)[-1])
    for name, fn in fns.items():
        x_fin, _ = reverse_sde_tracked(x_init, fn, times, rng_for("exp05-A-noise"))
        cov = np.cov(x_fin.T)
        rows.append({
            "sampler": name,
            "top_eigenvalue": float(np.linalg.eigvalsh(cov)[-1]),
            "top_eigenvalue_true": eig_true,
            "cov_frobenius_rel_error": float(
                np.linalg.norm(cov - sigma_true) / np.linalg.norm(sigma_true)
            ),
            "marginal_var": float(np.mean(np.diag(cov))),
        })
    write_csv(out / "setup3_modelA_terminal.csv", rows)

    fig, ax = new_figure()
    names = [r["sampler"] for r in rows]
    ax.bar(names, [r["top_eigenvalue"] for r in rows], color="C0", alpha=0.8)
    ax.axhline(eig_true, color="k", linestyle=":", label="true top eigenvalue")
    ax.set(ylabel="top eigenvalue of sample covariance",
           title=f"Model A (beta = {beta}): recovery of the global mode")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    save_figure(fig, out / "setup3_global_mode_recovery.png")


def main() -> None:
    args = experiment_parser("exp_05_reverse_dynamics", __doc__).parse_args()
    out = ensure_dir(args.output_dir)
    n_batch = 32 if args.quick else 200
    n_steps = 24 if args.quick else 80

    write_json(out / "params.json", {
        "n_sites": N_SITES, "rho": RHO, "t_max": T_MAX, "t_min": T_MIN,
        "n_batch": n_batch, "n_steps": n_steps,
        "sde_convention": "forward dX = -X dt + sqrt(2) dW; reverse "
                          "Euler-Maruyama x <- x + h(x + 2s) + sqrt(2h) xi",
        "provenance": provenance(),
    })

    setup1_laplace_generation(out, n_batch, n_steps)
    setup2_laplace_reconstruction(out, n_batch, n_steps)
    setup3_model_a(out, n_batch, n_steps)
    print(f"exp_05 done -> {out}")


if __name__ == "__main__":
    main()
