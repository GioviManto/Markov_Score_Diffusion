"""Experiment 09 -- Mixture-message closure: how many components beat a Gaussian?

This answers question R4 of `bp_markov_diffusion_gaussian_approx.pdf` ("add
Gaussian-mixture message approximations to see how many mixture components are
needed to beat single-Gaussian closure"), which is also F1 of the result ledger.

Why the design matters
----------------------
Audit finding F2 showed that for a linear-transition chain, moment-matched
single-Gaussian BP is *mathematically identical* to exact Gaussian BP on the
covariance-matched Gaussian model. So at C = 1 there is no such thing as a
"message error" distinct from a "model error" -- they are one number. The
distinction only becomes real for a message family rich enough to represent the
true message, which is exactly what a Gaussian mixture is.

To measure representation error and nothing else, Part 1 uses a prior whose
transition kernel is **exactly** a two-component Gaussian mixture
(`GaussianMixtureAR1`). The model is then exactly representable, no model error
exists at all, and every deviation from exact grid BP is attributable to the
message closure. Sweeping C therefore answers the report's question without
confounding.

Part 2 turns to a prior the mixture family cannot represent exactly
(`LaplaceAR1`, whose innovation is not a finite Gaussian mixture) and asks the
complementary question: with the model no longer exactly representable, does
adding message components still help, and where does it stop helping? The gap
between the C -> infinity limit and exact BP is model error; everything above it
is representation error. This is the first place in the project where the two
can be seen separately.

Part 3 is the bounded-support case (question R3). A uniform innovation fails a
Gaussian closure through *support* rather than *shape*: the true posterior is
exactly zero outside a finite interval and no Gaussian can represent that edge.
Its excess kurtosis (-1.2) places it between the kappa = 0.6 and kappa = 0.9
mixtures, so if |excess kurtosis| really is the one-number summary of closure
difficulty it should land between them.
"""

from __future__ import annotations

import time

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_gaussian import gaussian_chain_bp
from src.bp_grid import grid_bp_batch, make_grid
from src.bp_mixture import mixture_chain_bp
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.priors import GaussianMixtureAR1, LaplaceAR1, UniformAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 24
RHO = 0.85
GRID_A = 10.0
GRID_M = 1201  # reference grid: finer than the working default, this is the yardstick


def _mixture_components(prior: GaussianMixtureAR1):
    """The exact two-component innovation mixture of a GaussianMixtureAR1."""
    return (
        np.array([0.5, 0.5]),
        np.array([+prior.m, -prior.m]),
        np.array([prior.s**2, prior.s**2]),
    )


def _laplace_as_mixture(prior: LaplaceAR1, n_comp: int, rng):
    """Fit an n_comp Gaussian mixture to a Laplace innovation by EM on samples.

    A Laplace density is a scale mixture of Gaussians but not a *finite* one, so
    this is an approximation by construction -- which is the point of Part 2.
    """
    x = rng.laplace(0.0, prior.b, size=200_000)
    pi = np.full(n_comp, 1.0 / n_comp)
    mu = np.linspace(-2.0, 2.0, n_comp) * prior.b
    s2 = np.full(n_comp, prior.b**2)
    for _ in range(200):
        logp = (
            np.log(pi)[None, :]
            - 0.5 * np.log(2 * np.pi * s2)[None, :]
            - 0.5 * (x[:, None] - mu[None, :]) ** 2 / s2[None, :]
        )
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-12
        pi = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        s2 = np.maximum((r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, 1e-8)
    return pi, mu, s2


def _sweep(prior, pi, mu, s2, grid, weights, t_values, components, n_chains, tag):
    """Relative posterior-mean error of mixture BP vs exact grid BP."""
    rng = rng_for("exp09", tag)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    log_k = prior.log_transition_matrix(grid)

    rows = []
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        ref_norm = np.linalg.norm(m_ref)

        # The analytic single-Gaussian baseline, as an independent anchor for C=1.
        q = float(pi @ (s2 + mu**2) - (pi @ mu) ** 2)
        m_gauss = np.stack(
            [gaussian_chain_bp(x, prior.rho, q, alpha, delta).means for x in X]
        )
        gauss_err = float(np.linalg.norm(m_gauss - m_ref) / ref_norm)

        for c in components:
            t0 = time.perf_counter()
            m_mix = np.stack(
                [
                    mixture_chain_bp(x, prior.rho, pi, mu, s2, alpha, delta, c)[0]
                    for x in X
                ]
            )
            secs = time.perf_counter() - t0
            err = float(np.linalg.norm(m_mix - m_ref) / ref_norm)
            rows.append({
                "family": tag,
                "t": t,
                "n_components": c,
                "rel_mean_error": err,
                "gaussian_baseline_error": gauss_err,
                "improvement_over_gaussian": gauss_err / err if err > 0 else np.inf,
                "seconds_per_chain": secs / n_chains,
                "kernel_components": len(pi),
            })
    return rows


def part1_exact_family(grid, weights, cfg, out):
    """Pure representation error: the kernel is exactly representable."""
    rows = []
    for kappa in cfg["kappas"]:
        prior = GaussianMixtureAR1(RHO, kappa)
        pi, mu, s2 = _mixture_components(prior)
        rows += _sweep(
            prior, pi, mu, s2, grid, weights, cfg["t_values"],
            cfg["components"], cfg["n_chains"], f"gauss_mix_kappa{kappa:g}",
        )

    fig, ax = new_figure()
    for kappa in cfg["kappas"]:
        tag = f"gauss_mix_kappa{kappa:g}"
        sub = [r for r in rows if r["family"] == tag and r["t"] == cfg["t_values"][0]]
        sub.sort(key=lambda r: r["n_components"])
        ax.loglog([r["n_components"] for r in sub],
                  [r["rel_mean_error"] for r in sub],
                  "o-", label=rf"$\kappa={kappa}$")
    ax.set_xlabel("message components $C$")
    ax.set_ylabel("relative posterior-mean error")
    ax.set_title(f"Pure message-representation error at $t={cfg['t_values'][0]}$\n"
                 "(kernel exactly representable, so no model error)")
    ax.legend()
    save_figure(fig, out / "representation_error_vs_components.png")
    return rows


def part2_inexact_family(grid, weights, cfg, out):
    """Model error and representation error, separated for the first time."""
    prior = LaplaceAR1(RHO)
    rows = []
    for n_k in cfg["kernel_components"]:
        pi, mu, s2 = _laplace_as_mixture(prior, n_k, rng_for("exp09-lapfit", n_k))
        sub = _sweep(
            prior, pi, mu, s2, grid, weights, cfg["t_values"],
            cfg["components"], cfg["n_chains"], f"laplace_kernel{n_k}",
        )
        for r in sub:
            r["kernel_fit_components"] = n_k
        rows += sub

    fig, ax = new_figure()
    for n_k in cfg["kernel_components"]:
        sub = [r for r in rows if r.get("kernel_fit_components") == n_k
               and r["t"] == cfg["t_values"][0]]
        sub.sort(key=lambda r: r["n_components"])
        ax.loglog([r["n_components"] for r in sub],
                  [r["rel_mean_error"] for r in sub],
                  "o-", label=f"kernel fitted with {n_k} comps")
    ax.set_xlabel("message components $C$")
    ax.set_ylabel("relative posterior-mean error")
    ax.set_title(f"Laplace chain at $t={cfg['t_values'][0]}$: the floor is model error,\n"
                 "the decay above it is representation error")
    ax.legend(fontsize=8)
    save_figure(fig, out / "laplace_model_vs_representation.png")
    return rows


def part3_bounded_support(grid, weights, cfg, out):
    """Bounded support (R3): failure through support rather than shape."""
    prior = UniformAR1(RHO)
    rows = []
    for n_k in cfg["kernel_components"]:
        rng = rng_for("exp09-unifit", n_k)
        x = rng.uniform(-prior.half_width, prior.half_width, size=200_000)
        pi = np.full(n_k, 1.0 / n_k)
        mu = np.linspace(-1.0, 1.0, n_k) * prior.half_width
        s2 = np.full(n_k, (prior.half_width / max(n_k, 2)) ** 2 + 1e-3)
        for _ in range(200):
            logp = (
                np.log(pi)[None, :]
                - 0.5 * np.log(2 * np.pi * s2)[None, :]
                - 0.5 * (x[:, None] - mu[None, :]) ** 2 / s2[None, :]
            )
            logp -= logp.max(axis=1, keepdims=True)
            r = np.exp(logp)
            r /= r.sum(axis=1, keepdims=True)
            nk = r.sum(axis=0) + 1e-12
            pi = nk / len(x)
            mu = (r * x[:, None]).sum(axis=0) / nk
            s2 = np.maximum(
                (r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, 1e-8
            )
        sub = _sweep(
            prior, pi, mu, s2, grid, weights, cfg["t_values"],
            cfg["components"], cfg["n_chains"], f"uniform_kernel{n_k}",
        )
        for r_ in sub:
            r_["kernel_fit_components"] = n_k
        rows += sub
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_09_mixture_message_closure",
        "How many mixture components are needed to beat single-Gaussian closure?",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 601, "n_chains": 8, "t_values": (0.05, 0.4),
        "components": (1, 2, 4), "kappas": (0.6, 0.9), "kernel_components": (2, 4),
    }
    full = {
        "grid_size": GRID_M, "n_chains": 32,
        "t_values": (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
        "components": (1, 2, 3, 4, 6, 8, 12, 16),
        "kappas": (0.3, 0.6, 0.9), "kernel_components": (2, 4, 8),
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "exact_family": ("pure representation error (exactly representable kernel)",
                         lambda g, w, o: write_csv(o / "representation_error.csv",
                                                   part1_exact_family(g, w, cfg, o))),
        "inexact_family": ("Laplace: model vs representation error",
                           lambda g, w, o: write_csv(o / "laplace_closure.csv",
                                                     part2_inexact_family(g, w, cfg, o))),
        "bounded_support": ("uniform innovation (bounded support)",
                            lambda g, w, o: write_csv(o / "bounded_support.csv",
                                                      part3_bounded_support(g, w, cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, cfg["grid_size"])

    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "rho": RHO, "grid_half_width": GRID_A,
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg, **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(grid, weights, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
