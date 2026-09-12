"""Experiment 11 -- Is the locality law second-order universal? (question F5)

For the Gaussian chain the project already has an exact locality law (ledger
G12): the RMS error of the radius-r estimator -- the one that predicts a_i from
only x_{i-r..i+r} -- decays exactly as q^r, with

    q = (J_d - sqrt(J_d^2 - 4 beta^2)) / (2 |beta|),

an explicit function of rho and t through the posterior precision. It is the
statement behind B15, the reading that a CNN score head with receptive field
r >~ xi log(1/eps) is near-optimal.

F5 asks for the non-Gaussian analogue, "basis-independent". The sharp version of
that question, and the one this experiment tests, is:

    Is the locality decay rate a property of the second-order structure alone --
    so that every innovation law with the same (rho, q) shares the Gaussian
    rate -- or does it depend on the shape of the innovation?

The two outcomes mean quite different things. If the rate is second-order
universal, then B15's receptive-field prescription transfers to non-Gaussian
data unchanged, and the architecture reading is safe. If the rate depends on
shape, then heavy-tailed or bimodal data needs a different receptive field than
its covariance-matched Gaussian would suggest, and B15 is only a Gaussian
statement.

Method. For each prior and noise level, compute the exact posterior mean two
ways with grid BP: on the full chain, and on the window [i-r, i+r] treated as a
chain in its own right (which is exactly what an estimator with access to only
those observations can do). The decay of ||m^(r) - m^full|| in r is the locality
law. Fitting log-error against r gives the rate, which is compared against the
Gaussian chain at matched (rho, q).

Everything is measured at the centre site of the window to avoid contaminating
the result with edge effects.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import grid_bp_batch, make_grid
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import (
    GaussianAR1,
    GaussianMixtureAR1,
    LaplaceAR1,
    StudentTAR1,
    UniformAR1,
)
from src.stationary import (
    drifted_log_density,
    invariant_log_density,
    sample_stationary_batch,
)
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 41  # odd, so there is a well-defined centre site
CENTRE = N_SITES // 2
GRID_A = 8.0


def _families(rho: float):
    """Priors sharing the same (rho, q) but differing beyond second moments."""
    return [
        GaussianAR1(rho),
        LaplaceAR1(rho),
        StudentTAR1(rho, nu=5.0),
        UniformAR1(rho),
        GaussianMixtureAR1(rho, kappa=0.6),
        GaussianMixtureAR1(rho, kappa=0.9),
    ]


def locality_curve(prior, grid, weights, X, t, radii, log_mu_full=None, window_mu=None):
    """Error of the radius-r estimator at the centre site, against full-chain BP.

    `log_mu_full` is the initial law at site 1, for the full-chain reference. `window_mu` is
    a callable ``lo -> log_mu`` giving the initial law at a window's left endpoint, since
    that law depends on where the window starts. Both default to the standard normal that
    `grid_bp_batch` assumes, which reproduces the original measurement exactly.

    The window's initial law is the whole subtlety. A contiguous window of a Markov chain is
    itself a chain with the same kernel, so window BP returns the exact conditional
    expectation E[a_C | x_window] -- but only when it is handed the true marginal law of the
    window's *left endpoint*. Supplying N(0,1) there is right for the Gaussian chain at every
    site, and wrong for every other family at every site past the first.
    """
    alpha, delta = alpha_delta(t)
    log_k = prior.log_transition_matrix(grid)

    m_full, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta, log_mu_full)
    m_ref = m_full[:, CENTRE]

    out = []
    for r in radii:
        lo, hi = CENTRE - r, CENTRE + r + 1
        mu_lo = None if window_mu is None else window_mu(lo)
        m_win, _ = grid_bp_batch(grid, weights, log_k, X[:, lo:hi], alpha, delta, mu_lo)
        err = float(np.sqrt(np.mean((m_win[:, r] - m_ref) ** 2)))
        out.append((r, err))
    return out


def _fit_rate(radii, errors):
    """Geometric decay rate q from a log-linear fit, with its standard error."""
    r = np.asarray(radii, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = e > 0
    if ok.sum() < 3:
        return np.nan, np.nan
    r, e = r[ok], np.log(e[ok])
    A = np.vstack([r, np.ones(len(r))]).T
    beta, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ beta
    s2 = resid @ resid / max(len(r) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(np.exp(beta[0])), float(np.exp(beta[0]) * np.sqrt(cov[0, 0]))


def part1_universality(cfg, out):
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []
    for rho in cfg["rhos"]:
        for prior in _families(rho):
            rng = rng_for("exp11", prior.name, rho)
            A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
            for t in cfg["t_values"]:
                alpha, delta = alpha_delta(t)
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                curve = locality_curve(prior, grid, weights, X, t, cfg["radii"])
                q_hat, q_se = _fit_rate(*zip(*curve))
                for r, err in curve:
                    rows.append({
                        "family": prior.name,
                        "excess_kurtosis": prior.innovation_excess_kurtosis,
                        "rho": rho,
                        "t": t,
                        "radius": r,
                        "rms_error": err,
                        "fitted_rate": q_hat,
                        "fitted_rate_se": q_se,
                    })

    # Ratio of each family's rate to the Gaussian rate at matched (rho, t).
    for row in rows:
        ref = next(
            x["fitted_rate"] for x in rows
            if x["family"] == "gaussian" and x["rho"] == row["rho"]
            and x["t"] == row["t"]
        )
        row["rate_over_gaussian"] = row["fitted_rate"] / ref if ref > 0 else np.nan

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    rho0, t0 = cfg["rhos"][0], cfg["t_values"][len(cfg["t_values"]) // 2]
    for prior in _families(rho0):
        sub = sorted(
            [r for r in rows if r["family"] == prior.name
             and r["rho"] == rho0 and r["t"] == t0],
            key=lambda r: r["radius"],
        )
        ax[0].semilogy([r["radius"] for r in sub], [r["rms_error"] for r in sub],
                       "o-", label=prior.name, ms=3)
    ax[0].set_xlabel("window radius $r$")
    ax[0].set_ylabel("RMS error at the centre site")
    ax[0].set_title(rf"Locality decay, $\rho={rho0}$, $t={t0}$")
    ax[0].legend(fontsize=7)

    for prior in _families(rho0):
        sub = [r for r in rows if r["family"] == prior.name and r["rho"] == rho0
               and r["radius"] == cfg["radii"][0]]
        sub.sort(key=lambda r: r["t"])
        ax[1].semilogx([r["t"] for r in sub], [r["rate_over_gaussian"] for r in sub],
                       "o-", label=prior.name, ms=3)
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("noise level $t$")
    ax[1].set_ylabel("fitted rate / Gaussian rate")
    ax[1].set_title("Is the rate second-order universal?")
    ax[1].legend(fontsize=7)
    save_figure(fig, out / "locality_universality.png")
    return rows


def _rate_ratios(rows, key="init"):
    """Attach each row's fitted rate divided by the Gaussian rate at matched conditions.

    Grouped by `key` as well as (rho, t), so an arm is only ever normalised against the
    Gaussian chain measured under the *same* protocol. Normalising across arms would mix the
    thing being measured with the thing being corrected.
    """
    for row in rows:
        ref = next(
            x["fitted_rate"] for x in rows
            if x["family"] == "gaussian" and x["rho"] == row["rho"]
            and x["t"] == row["t"] and x[key] == row[key]
        )
        row["rate_over_gaussian"] = row["fitted_rate"] / ref if ref > 0 else np.nan


def part2_stationary(cfg, out):
    """The locality curve with the window's initial law made correct, three ways.

    Part 1 draws chains from ``a_1 ~ N(0,1)`` and hands every window BP the same N(0,1) as
    its initial law. For a window ``[C-r, C+r]`` that is the marginal law of site ``C-r``
    only for the Gaussian chain; for the others site ``C-r`` has drifted towards the
    invariant law. So part 1's estimator is the exact conditional expectation for the family
    that everything else is normalised against, and an approximation for the families whose
    departure from it is the reported result. The measured 1.12-1.46x rate ratios therefore
    contain an unknown amount of initial-law mismatch.

    Three arms separate the two effects rather than assuming which dominates:

    ``n01``       the committed protocol -- N(0,1) data, N(0,1) window law. Reproduces
                  part 1 and exists so the other arms have something to be compared against.
    ``n01_exact`` the same data, but each window given the *exact* marginal of its own left
                  endpoint, computed by pushing N(0,1) through the kernel that many times.
                  The difference from ``n01`` is the error in the committed numbers, on the
                  committed data, with nothing else changed.
    ``invariant`` strictly stationary chains with the invariant law as initial law
                  everywhere. Here the window estimator is exact at every site, so this is
                  the locality law free of the artefact -- a T3 quantity, not a proxy.

    The middle arm is what makes this a measurement instead of a replacement: without it we
    would know the new numbers but not how wrong the old ones were.
    """
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    rows = []
    for rho in cfg["rhos"]:
        for prior in _families(rho):
            inv = invariant_log_density(prior, grid, weights)
            # Marginal at each site the windows actually start from, computed once.
            drifted = {
                CENTRE - r: drifted_log_density(prior, grid, weights, CENTRE - r)
                for r in cfg["radii"]
            }
            arms = {
                "n01": (None, None),
                "n01_exact": (None, drifted.get),
                "invariant": (inv.log_density, lambda _lo: inv.log_density),
            }

            for init, (log_mu_full, window_mu) in arms.items():
                rng = rng_for("exp11-stat", prior.name, rho, init)
                if init == "invariant":
                    A = sample_stationary_batch(prior, rng, cfg["n_chains"], N_SITES)
                else:
                    A = np.stack(
                        [prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])]
                    )

                for t in cfg["t_values"]:
                    alpha, delta = alpha_delta(t)
                    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                    curve = locality_curve(prior, grid, weights, X, t, cfg["radii"],
                                           log_mu_full, window_mu)
                    q_hat, q_se = _fit_rate(*zip(*curve))
                    for r, err in curve:
                        rows.append({
                            "init": init,
                            "family": prior.name,
                            "excess_kurtosis": prior.innovation_excess_kurtosis,
                            "rho": rho,
                            "t": t,
                            "radius": r,
                            "rms_error": err,
                            "fitted_rate": q_hat,
                            "fitted_rate_se": q_se,
                            "invariant_iters": inv.n_iter,
                        })
                    print(f"  {init:10s} {prior.name:18s} rho={rho} t={t:5.2f} "
                          f"rate={q_hat:.4f}+-{q_se:.4f}", flush=True)

    _rate_ratios(rows)

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    rho0 = cfg["rhos"][len(cfg["rhos"]) // 2]
    styles = {"n01": ("--", 0.55), "n01_exact": (":", 0.75), "invariant": ("-", 1.0)}
    for prior in _families(rho0):
        for init, (ls, alpha_) in styles.items():
            sub = [r for r in rows if r["family"] == prior.name and r["rho"] == rho0
                   and r["init"] == init and r["radius"] == cfg["radii"][0]]
            sub.sort(key=lambda r: r["t"])
            if not sub:
                continue
            ax[0].semilogx([r["t"] for r in sub], [r["rate_over_gaussian"] for r in sub],
                           ls, alpha=alpha_, label=f"{prior.name} / {init}", lw=1.4)
    ax[0].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].set_xlabel("noise level $t$")
    ax[0].set_ylabel("fitted rate / Gaussian rate")
    ax[0].set_title(rf"Rate ratio by protocol, $\rho={rho0}$")
    ax[0].legend(fontsize=5, ncol=2)

    # How much the committed protocol was off by, per family: the n01 ratio against the
    # strictly stationary one. Unity means the artefact did not matter.
    for prior in _families(rho0):
        ts, rel = [], []
        for t in cfg["t_values"]:
            def pick(init):
                return [r["rate_over_gaussian"] for r in rows
                        if r["family"] == prior.name and r["rho"] == rho0
                        and r["init"] == init and r["t"] == t and r["radius"] == cfg["radii"][0]]
            a, b = pick("n01"), pick("invariant")
            if a and b and b[0] != 0:
                ts.append(t)
                rel.append(a[0] / b[0])
        if ts:
            ax[1].semilogx(ts, rel, "o-", ms=3, label=prior.name)
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("noise level $t$")
    ax[1].set_ylabel("committed ratio / stationary ratio")
    ax[1].set_title("How much of the effect was the initial law?")
    ax[1].legend(fontsize=7)
    save_figure(fig, out / "locality_stationary.png")
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_11_nongaussian_locality",
        "Does the Gaussian locality law survive beyond second order?",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 301, "n_chains": 32, "rhos": (0.85,),
        "t_values": (0.1, 0.4), "radii": (1, 2, 3, 4, 5),
    }
    full = {
        "grid_size": 401, "n_chains": 128, "rhos": (0.5, 0.85, 0.95),
        "t_values": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
        "radii": (1, 2, 3, 4, 5, 6, 7, 8),
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "universality": ("locality rate vs innovation shape",
                         lambda o: write_csv(o / "locality.csv",
                                             part1_universality(cfg, o))),
        "stationary": ("locality rate with the window's initial law made correct",
                       lambda o: write_csv(o / "locality_stationary.csv",
                                           part2_stationary(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "centre": CENTRE, "grid_half_width": GRID_A,
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
