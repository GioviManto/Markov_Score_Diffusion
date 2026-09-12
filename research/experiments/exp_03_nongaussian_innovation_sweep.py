"""Experiment 03 -- When does the Gaussian score baseline fail? (Layer 3 sweep)

Innovation families, all with Cov(a_i, a_j) = rho^{|i-j|} (identical second-order
structure), differing only beyond second moments:

    laplace              excess kurtosis +3        (heavy tails)
    student_t nu=5       excess kurtosis +6        (heavier tails)
    student_t nu=8       excess kurtosis +1.5      (mild tails)
    gauss_mix kappa=0.3  excess kurtosis -0.18     (mildly bimodal)
    gauss_mix kappa=0.6  excess kurtosis -0.72     (bimodal)
    gauss_mix kappa=0.9  excess kurtosis -1.62     (strongly bimodal)

For each (family, rho, t): fine-grid BP reference versus the analytic Gaussian
baseline. By the equivalence proposition the Gaussian baseline is
simultaneously (i) single-Gaussian moment-matched message BP and (ii) exact BP
under the covariance-matched Gaussian AR(1) prior -- so this sweep measures the
message-level and model-level Gaussian error at once (they coincide at this
level; richer message families are required to separate them).

Outputs: sweep CSV, error-vs-t curves per family, error-vs-kurtosis plot,
and a "when Gaussian works" summary table (markdown).
"""

from __future__ import annotations

import numpy as np

from common import experiment_parser, provenance
from src.bp_gaussian import gaussian_chain_bp
from src.bp_grid import grid_bp, make_grid, score_from_posterior_mean
from src.metrics import cosine, max_abs, mse, rel_l2
from src.noising import log_likelihood_matrix, ou_noise_sample
from src.plotting import new_figure, save_figure
from src.priors import GaussianMixtureAR1, LaplaceAR1, StudentTAR1
from src.stationary import invariant_log_density, sample_stationary
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites
T_VALUES = tuple(FROZEN.t_grid)
RHOS = (0.5, 0.85, 0.95)


def families(rho: float):
    return [
        LaplaceAR1(rho),
        StudentTAR1(rho, nu=5.0),
        StudentTAR1(rho, nu=8.0),
        GaussianMixtureAR1(rho, kappa=0.3),
        GaussianMixtureAR1(rho, kappa=0.6),
        GaussianMixtureAR1(rho, kappa=0.9),
    ]


def main() -> None:
    args = experiment_parser("exp_03_nongaussian_innovation_sweep", __doc__).parse_args()
    if args.list_parts:
        print("baseline\nstationary")
        return
    out = ensure_dir(args.output_dir)
    n_trials = 3 if args.quick else 12
    m_ref = 401 if args.quick else 801
    t_values = T_VALUES[1::2] if args.quick else T_VALUES
    rhos = (0.85,) if args.quick else RHOS

    if args.only and args.only.strip() == "stationary":
        run_stationary(out, rhos, t_values, n_trials, m_ref, args)
        return

    write_json(out / "params.json", {
        "n_sites": N_SITES, "rhos": rhos, "t_values": t_values,
        "n_trials": n_trials, "ref_grid": {"M": m_ref, "A": 8.0},
        "provenance": provenance(),
    })

    rows: list[dict] = []
    grid, w = make_grid(8.0, m_ref)

    for rho in rhos:
        for prior in families(rho):
            log_K = prior.log_transition_matrix(grid)
            for t in t_values:
                per_trial = []
                for trial in range(n_trials):
                    rng = rng_for("exp03", prior.name, rho, t, trial)
                    a = prior.sample(rng, N_SITES)
                    x, alpha, delta = ou_noise_sample(rng, a, t)

                    ref = grid_bp(
                        grid, w, log_K,
                        log_likelihood_matrix(grid, x, alpha, delta),
                        x, alpha, delta,
                    )
                    s_ref = score_from_posterior_mean(x, ref.means, alpha, delta)
                    gauss = gaussian_chain_bp(x, rho, prior.q, alpha, delta)
                    s_gauss = score_from_posterior_mean(x, gauss.means, alpha, delta)

                    per_trial.append({
                        "mean_mse": mse(gauss.means, ref.means),
                        "mean_rel": rel_l2(gauss.means, ref.means),
                        "score_mse": mse(s_gauss, s_ref),
                        "score_rel": rel_l2(s_gauss, s_ref),
                        "cos": cosine(s_gauss, s_ref),
                        "max_abs": max_abs(s_gauss, s_ref),
                    })

                def med(k):  # noqa: B023
                    return float(np.median([p[k] for p in per_trial]))

                def q90(k):  # noqa: B023
                    return float(np.quantile([p[k] for p in per_trial], 0.9))

                rows.append({
                    "family": prior.name,
                    "rho": rho,
                    "t": t,
                    "excess_kurtosis": prior.innovation_excess_kurtosis,
                    "posterior_mean_mse_median": med("mean_mse"),
                    "posterior_mean_rel_median": med("mean_rel"),
                    "score_mse_median": med("score_mse"),
                    "score_rel_median": med("score_rel"),
                    "score_rel_q90": q90("score_rel"),
                    "score_cosine_median": med("cos"),
                    "score_max_abs_median": med("max_abs"),
                })

    write_csv(out / "innovation_sweep.csv", rows)
    make_figures(rows, rhos, t_values, out)
    write_summary_table(rows, rhos, out)
    print(f"exp_03 done -> {out}")


def run_stationary(out, rhos, t_values, n_trials, m_ref, args) -> None:
    """The same sweep on strictly stationary chains, testing a prediction the note makes.

    The note states that sampling from the invariant law instead of starting at N(0,1) would
    not change the closure results materially, and offers no run behind it. The reasoning is
    that the Gaussian baseline depends on the prior only through its second moments, and
    those are identical under both initialisations: Var(a_i) = 1 and Cov(a_i, a_j) =
    rho^|i-j| hold exactly either way, since only the *shape* of the marginal drifts. So the
    baseline arm is literally unchanged and only the reference arm can move.

    That makes this a real prediction with a way to be wrong. If the closure error shifts
    appreciably, the marginal shape at early sites was doing more work than the argument
    allows, and the family comparison would need restating on stationary data throughout.

    Both the data and the reference BP's initial law change together -- a strictly stationary
    chain whose BP is told the law is N(0,1) would be neither construction.
    """
    grid, w = make_grid(8.0, m_ref)
    write_json(out / "params_stationary.json", {
        "n_sites": N_SITES, "rhos": rhos, "t_values": t_values,
        "n_trials": n_trials, "ref_grid": {"M": m_ref, "A": 8.0},
        "init": "invariant", "provenance": provenance(),
    })

    rows: list[dict] = []
    for rho in rhos:
        for prior in families(rho):
            log_K = prior.log_transition_matrix(grid)
            inv = invariant_log_density(prior, grid, w)
            for t in t_values:
                per_trial = []
                for trial in range(n_trials):
                    rng = rng_for("exp03-stat", prior.name, rho, t, trial)
                    a = sample_stationary(prior, rng, N_SITES)
                    x, alpha, delta = ou_noise_sample(rng, a, t)

                    ref = grid_bp(
                        grid, w, log_K,
                        log_likelihood_matrix(grid, x, alpha, delta),
                        x, alpha, delta, log_mu=inv.log_density,
                    )
                    s_ref = score_from_posterior_mean(x, ref.means, alpha, delta)
                    # Unchanged from the baseline arm on purpose: the Gaussian model sees
                    # only (rho, q), and neither moves under a change of initial law.
                    gauss = gaussian_chain_bp(x, rho, prior.q, alpha, delta)
                    s_gauss = score_from_posterior_mean(x, gauss.means, alpha, delta)

                    per_trial.append({
                        "mean_mse": mse(gauss.means, ref.means),
                        "mean_rel": rel_l2(gauss.means, ref.means),
                        "score_mse": mse(s_gauss, s_ref),
                        "score_rel": rel_l2(s_gauss, s_ref),
                        "cos": cosine(s_gauss, s_ref),
                        "max_abs": max_abs(s_gauss, s_ref),
                    })

                def med(k):  # noqa: B023
                    return float(np.median([p[k] for p in per_trial]))

                rows.append({
                    "init": "invariant",
                    "family": prior.name,
                    "rho": rho,
                    "t": t,
                    "excess_kurtosis": prior.innovation_excess_kurtosis,
                    "invariant_iters": inv.n_iter,
                    "posterior_mean_mse_median": med("mean_mse"),
                    "posterior_mean_rel_median": med("mean_rel"),
                    "score_mse_median": med("score_mse"),
                    "score_rel_median": med("score_rel"),
                    "score_cosine_median": med("cos"),
                    "score_max_abs_median": med("max_abs"),
                })
                print(f"  {prior.name:18s} rho={rho} t={t:5.2f} "
                      f"score_rel={rows[-1]['score_rel_median']:.5f}", flush=True)

    write_csv(out / "innovation_sweep_stationary.csv", rows)
    print(f"exp_03 stationary arm done -> {out}")


def make_figures(rows, rhos, t_values, out) -> None:
    rho_mid = 0.85 if 0.85 in rhos else rhos[0]

    fig, ax = new_figure()
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam and r["rho"] == rho_mid]
        ax.plot([r["t"] for r in sub], [r["score_rel_median"] for r in sub],
                marker="o", label=fam)
    ax.set(xscale="log", yscale="log", xlabel="diffusion time t",
           ylabel="median relative score error",
           title=f"Gaussian baseline error by innovation family (rho = {rho_mid})")
    ax.legend(fontsize=8)
    save_figure(fig, out / "sweep_error_vs_t.png")

    # Error vs rho at fixed moderate t.
    t_mid = 0.2 if 0.2 in t_values else t_values[len(t_values) // 2]
    fig, ax = new_figure()
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam and r["t"] == t_mid]
        ax.plot([r["rho"] for r in sub], [r["score_rel_median"] for r in sub],
                marker="s", label=fam)
    ax.set(yscale="log", xlabel="correlation rho",
           ylabel="median relative score error",
           title=f"Gaussian baseline error vs correlation (t = {t_mid})")
    ax.legend(fontsize=8)
    save_figure(fig, out / "sweep_error_vs_rho.png")

    # Error vs non-Gaussianity (finite-kurtosis families) at small and large t.
    fig, ax = new_figure()
    for t_sel, marker in [(min(t_values), "o"), (t_mid, "s"), (max(t_values), "^")]:
        sub = [r for r in rows
               if r["rho"] == rho_mid and r["t"] == t_sel
               and np.isfinite(r["excess_kurtosis"])]
        sub.sort(key=lambda r: r["excess_kurtosis"])
        ax.plot([r["excess_kurtosis"] for r in sub],
                [r["score_rel_median"] for r in sub],
                marker=marker, linestyle="--", label=f"t = {t_sel}")
    ax.set(yscale="log", xlabel="innovation excess kurtosis",
           ylabel="median relative score error",
           title=f"Error vs non-Gaussianity (rho = {rho_mid}); "
                 "negative = bimodal, positive = heavy tails")
    ax.legend()
    save_figure(fig, out / "sweep_error_vs_kurtosis.png")


def write_summary_table(rows, rhos, out) -> None:
    lines = [
        "# When does the Gaussian score baseline work?",
        "",
        "Median relative score error of the analytic Gaussian baseline "
        "(== covariance-matched Gaussian model) vs fine-grid BP.",
        "",
        "| family | kurt. | rho | small t (min) | mid t (0.2) | large t (max) |",
        "|---|---|---|---|---|---|",
    ]
    t_all = sorted({r["t"] for r in rows})
    t_small, t_large = t_all[0], t_all[-1]
    t_mid = 0.2 if 0.2 in t_all else t_all[len(t_all) // 2]
    for rho in rhos:
        for fam in sorted({r["family"] for r in rows}):
            def get(t_sel):  # noqa: B023
                for r in rows:
                    if r["family"] == fam and r["rho"] == rho and r["t"] == t_sel:
                        return r
                return None
            r_s, r_m, r_l = get(t_small), get(t_mid), get(t_large)
            if r_s is None:
                continue
            kurt = r_s["excess_kurtosis"]
            kurt_str = f"{kurt:+.2f}" if np.isfinite(kurt) else "inf"
            lines.append(
                f"| {fam} | {kurt_str} | {rho} "
                f"| {r_s['score_rel_median']:.3f} "
                f"| {r_m['score_rel_median']:.3f} "
                f"| {r_l['score_rel_median']:.3f} |"
            )
    (out / "SUMMARY_TABLE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
