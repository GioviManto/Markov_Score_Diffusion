"""Experiment 02 -- Gaussian-message error on the Laplace chain, done right (Layer 3).

Three parts:

(A) Artifact demonstration (audit finding F1): the legacy grid-projected
    Gaussian BP versus the analytic information-form Gaussian BP. The two
    implement the same projection map, so any disagreement is numerical. At
    weakly informative t the legacy version collapses onto the grid boundary;
    the analytic version cannot.

(B) Corrected Gaussian-closure error: analytic Gaussian BP (== exact BP on the
    covariance-matched Gaussian AR(1) model, by the equivalence proposition)
    versus fine-grid reference BP on the Laplace chain, per-trial distribution
    over many trials: median, quantile band, failure probability, worst case.

(C) Error-budget comparison: grid discretization error (M = 401 vs 801
    self-convergence) versus Gaussian closure error on identical data --
    establishing that the reference is orders of magnitude more accurate than
    the effect being measured.

Every score is produced through the posterior-mean identity, and the exact
relation s_hat - s_ref = (alpha/Delta)(m_hat - m_ref) is verified per trial.
"""

from __future__ import annotations

import numpy as np

from common import experiment_parser, provenance
from src.bp_gaussian import gaussian_chain_bp, grid_projected_gaussian_bp
from src.bp_grid import grid_bp, make_grid, score_from_posterior_mean
from src.metrics import cosine, max_abs, mse, rel_l2, score_mean_identity_residual
from src.noising import log_likelihood_matrix, ou_noise_sample
from src.plotting import new_figure, save_figure
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites
RHO = FROZEN.rho
T_VALUES = tuple(FROZEN.t_grid)


def main() -> None:
    args = experiment_parser("exp_02_laplace_gaussian_message_error", __doc__).parse_args()
    out = ensure_dir(args.output_dir)
    n_trials = 8 if args.quick else 60
    m_ref = 401 if args.quick else 801
    t_values = T_VALUES[1::3] if args.quick else T_VALUES

    prior = LaplaceAR1(RHO)
    write_json(out / "params.json", {
        "n_sites": N_SITES, "rho": RHO, "t_values": t_values,
        "n_trials": n_trials, "ref_grid": {"M": m_ref, "A": 8.0},
        "provenance": provenance(),
    })

    grid_ref, w_ref = make_grid(8.0, m_ref)
    grid_conv, w_conv = make_grid(8.0, 401)
    log_K_ref = prior.log_transition_matrix(grid_ref)
    log_K_conv = prior.log_transition_matrix(grid_conv)

    per_trial_rows: list[dict] = []
    artifact_rows: list[dict] = []
    worst: dict | None = None

    for t in t_values:
        for trial in range(n_trials):
            rng = rng_for("exp02", RHO, t, trial)
            a = prior.sample(rng, N_SITES)
            x, alpha, delta = ou_noise_sample(rng, a, t)

            # Reference: fine-grid BP (exact up to grid error, quantified in C).
            ref = grid_bp(
                grid_ref, w_ref, log_K_ref,
                log_likelihood_matrix(grid_ref, x, alpha, delta), x, alpha, delta,
            )
            s_ref = score_from_posterior_mean(x, ref.means, alpha, delta)

            # Corrected Gaussian baseline: analytic information-form BP.
            gauss = gaussian_chain_bp(x, RHO, prior.q, alpha, delta)
            s_gauss = score_from_posterior_mean(x, gauss.means, alpha, delta)

            # Legacy grid projection (artifact demo, part A).
            m_legacy, _ = grid_projected_gaussian_bp(
                grid_conv, w_conv, log_K_conv,
                log_likelihood_matrix(grid_conv, x, alpha, delta),
            )

            # Grid self-convergence on the same data (part C).
            conv = grid_bp(
                grid_conv, w_conv, log_K_conv,
                log_likelihood_matrix(grid_conv, x, alpha, delta), x, alpha, delta,
            )
            s_conv = score_from_posterior_mean(x, conv.means, alpha, delta)

            rel_err = rel_l2(s_gauss, s_ref)
            row = {
                "t": t, "trial": trial, "alpha": alpha, "delta": delta,
                "posterior_mean_mse": mse(gauss.means, ref.means),
                "posterior_mean_rel_error": rel_l2(gauss.means, ref.means),
                "score_mse": mse(s_gauss, s_ref),
                "score_rel_error": rel_err,
                "score_cosine": cosine(s_gauss, s_ref),
                "score_max_abs_error": max_abs(s_gauss, s_ref),
                "identity_residual": score_mean_identity_residual(
                    s_gauss, s_ref, gauss.means, ref.means, alpha, delta
                ),
                "grid_self_conv_rel_error": rel_l2(s_conv, s_ref),
                "legacy_vs_analytic_max_abs_mean": max_abs(m_legacy, gauss.means),
            }
            per_trial_rows.append(row)
            artifact_rows.append({
                "t": t, "trial": trial,
                "legacy_vs_analytic_max_abs_mean": row["legacy_vs_analytic_max_abs_mean"],
            })
            if worst is None or rel_err > worst["rel_err"]:
                i_bad = int(np.argmax(np.abs(gauss.means - ref.means)))
                worst = {
                    "rel_err": rel_err, "t": t, "trial": trial, "site": i_bad,
                    "grid": grid_ref.copy(), "belief": ref.beliefs[i_bad].copy(),
                    "mean_ref": float(ref.means[i_bad]),
                    "mean_gauss": float(gauss.means[i_bad]),
                    "var_gauss": float(gauss.variances[i_bad]),
                }

    write_csv(out / "laplace_per_trial.csv", per_trial_rows)
    summary = summarize(per_trial_rows, t_values)
    write_csv(out / "laplace_summary.csv", summary)
    make_figures(per_trial_rows, summary, worst, out)
    print(f"exp_02 done -> {out}")


def summarize(rows: list[dict], t_values) -> list[dict]:
    out = []
    for t in t_values:
        sub = [r for r in rows if r["t"] == t]
        rel = np.array([r["score_rel_error"] for r in sub])
        conv = np.array([r["grid_self_conv_rel_error"] for r in sub])
        legacy = np.array([r["legacy_vs_analytic_max_abs_mean"] for r in sub])
        out.append({
            "t": t,
            "rel_score_error_median": float(np.median(rel)),
            "rel_score_error_q10": float(np.quantile(rel, 0.10)),
            "rel_score_error_q90": float(np.quantile(rel, 0.90)),
            "rel_score_error_max": float(rel.max()),
            "p_rel_error_gt_0p1": float(np.mean(rel > 0.1)),
            "cosine_median": float(np.median([r["score_cosine"] for r in sub])),
            "posterior_mean_mse_median": float(
                np.median([r["posterior_mean_mse"] for r in sub])
            ),
            "grid_self_conv_median": float(np.median(conv)),
            "legacy_artifact_median": float(np.median(legacy)),
            "legacy_artifact_max": float(legacy.max()),
            "identity_residual_max": float(
                max(r["identity_residual"] for r in sub)
            ),
        })
    return out


def make_figures(rows, summary, worst, out) -> None:
    ts = [r["t"] for r in summary]

    # Corrected closure error with quantile band vs grid self-convergence.
    fig, ax = new_figure()
    med = [r["rel_score_error_median"] for r in summary]
    q10 = [r["rel_score_error_q10"] for r in summary]
    q90 = [r["rel_score_error_q90"] for r in summary]
    ax.fill_between(ts, q10, q90, alpha=0.25, label="closure error, 10-90%")
    ax.plot(ts, med, marker="o", label="Gaussian closure error (median)")
    ax.plot(ts, [r["grid_self_conv_median"] for r in summary], marker="s",
            linestyle="--", label="grid self-convergence 401 vs 801 (median)")
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="relative score error vs fine-grid BP",
           title=f"Laplace chain (rho = {RHO}): closure error vs grid error budget")
    ax.legend()
    save_figure(fig, out / "laplace_closure_vs_grid_budget.png")

    # Failure probability and cosine.
    fig, ax = new_figure()
    ax.plot(ts, [r["p_rel_error_gt_0p1"] for r in summary], marker="o")
    ax.set(xscale="log", xlabel="diffusion time t",
           ylabel="P(relative score error > 0.1)",
           title="Failure probability of the Gaussian score baseline")
    save_figure(fig, out / "laplace_failure_probability.png")

    # Legacy artifact size vs t.
    fig, ax = new_figure()
    ax.plot(ts, [r["legacy_artifact_median"] for r in summary], marker="o",
            label="median")
    ax.plot(ts, [r["legacy_artifact_max"] for r in summary], marker="^",
            linestyle="--", label="max")
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="max |legacy - analytic| posterior mean",
           title="Boundary-collapse artifact of grid-projected Gaussian BP")
    ax.legend()
    save_figure(fig, out / "legacy_artifact_size.png")

    # Worst-case belief.
    if worst is not None:
        fig, ax = new_figure()
        ax.plot(worst["grid"], worst["belief"], label="reference grid-BP belief")
        g = worst["grid"]
        gauss_b = np.exp(-0.5 * (g - worst["mean_gauss"]) ** 2 / worst["var_gauss"])
        gauss_b /= gauss_b.max() / worst["belief"].max()
        ax.plot(g, gauss_b, linestyle="--", label="Gaussian-BP belief (rescaled)")
        ax.axvline(worst["mean_ref"], color="C0", linestyle=":", label="ref mean")
        ax.axvline(worst["mean_gauss"], color="C1", linestyle=":", label="Gauss mean")
        ax.set(xlabel=f"a_{worst['site']}", ylabel="belief density",
               title=(f"Worst trial: t = {worst['t']:g}, rel. score error = "
                      f"{worst['rel_err']:.2f}"))
        ax.legend()
        save_figure(fig, out / "laplace_worst_case_belief.png")


if __name__ == "__main__":
    main()
