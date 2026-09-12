"""Experiment 10 -- Discrete alphabet: the headline retested with zero grid error.

Question F2 of the result ledger. Every result in this package so far has been
measured through a grid representation of continuous messages, and the grid
brings a tail of caveats: trapezoidal quadrature error, truncation, a resolution
condition at small t, and the ratio-lattice quantization that biases the Laplace
M-step no matter how fine the grid gets.

On a finite alphabet all of that vanishes at once. The latent chain takes values
in a finite set of levels, the noising is still the continuous OU channel, and
the messages are S-vectors -- so BP is exact up to roundoff and EM is literally
Baum-Welch. The prior is strongly non-Gaussian by construction, so this is not a
retreat to an easy case.

Part 1 (recovery and rate). Baum-Welch from noisy observations only: transition
  matrix error against the number of observed chains, and the exactness of the
  monotonicity guarantee with no numerical slack anywhere.

Part 2 (the headline, confound-free). EM-BP versus a denoising-score-matching
  network on the discrete chain. The reference denoiser is *exact* here rather
  than a fine-grid approximation, so any margin cannot be a grid artifact.

Part 3 (alphabet size). How the comparison moves as the alphabet grows: the
  number of learned parameters is S(S-1), so this sweeps the structured method
  from far fewer parameters than the network to comparably many.
"""

from __future__ import annotations

import time

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.denoiser import dsm_posterior_mean, evaluate_denoiser, train_dsm_denoiser
from src.discrete import (
    discrete_bp,
    fit_em_discrete,
    make_random_chain,
    monotone_violation,
)
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.protocols import one_view_groups as noisy_groups
from src.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES = 32
T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)
N_TEST = 256
PARAMETERIZATIONS = ("eps", "x0")




def make_test_set(chain, t_values, n_test):
    """Held-out chains plus the EXACT reference denoiser at each level."""
    rng = rng_for("exp10-test", chain.n_states)
    A = np.stack([chain.sample(rng, N_SITES) for _ in range(n_test)])
    bundle = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        res = discrete_bp(chain.levels, chain.K, X, alpha, delta, chain.mu)
        bundle[t] = (X, res.means)
    return A, bundle


def part1_recovery(cfg, out):
    rows = []
    for n_states in cfg["alphabet_sizes"]:
        chain = make_random_chain(
            n_states, rng_for("exp10-chain", n_states), concentration=0.6
        )
        for n_chains in cfg["sizes"]:
            errs = []
            for rep in range(cfg["n_rep"]):
                rng = rng_for("exp10-p1", n_states, n_chains, rep)
                A = np.stack([chain.sample(rng, N_SITES) for _ in range(n_chains)])
                k0 = rng.dirichlet(np.ones(n_states), size=n_states).T
                k_fit, trace = fit_em_discrete(
                    chain.levels, k0, noisy_groups(A, T_TRAIN, rng), n_iters=300
                )
                errs.append(np.abs(k_fit - chain.K).max())
                viol = monotone_violation(trace["log_evidence"])
            rows.append({
                "n_states": n_states,
                "n_chains": n_chains,
                "n_rep": cfg["n_rep"],
                "K_max_error_rmse": float(np.sqrt(np.mean(np.square(errs)))),
                "K_max_error_median": float(np.median(errs)),
                "monotone_violation_last": float(viol),
                "n_params": n_states * (n_states - 1),
            })

    fig, ax = new_figure()
    ns = np.array(cfg["sizes"], dtype=float)
    for n_states in cfg["alphabet_sizes"]:
        sub = [r for r in rows if r["n_states"] == n_states]
        ax.loglog([r["n_chains"] for r in sub], [r["K_max_error_rmse"] for r in sub],
                  "o-", label=f"$S={n_states}$")
    ax.loglog(ns, rows[0]["K_max_error_rmse"] * np.sqrt(ns[0] / ns), "k--", lw=1,
              label=r"$N^{-1/2}$")
    ax.set_xlabel("number of observed noisy chains $N$")
    ax.set_ylabel(r"$\max|\hat K - K|$")
    ax.set_title("Baum-Welch recovery through the OU channel (no grid anywhere)")
    ax.legend()
    save_figure(fig, out / "discrete_recovery.png")
    return rows


def _compare(chain, cfg, n_chains, out_rows, tag_extra=None):
    """EM-BP vs both network parameterizations at one (chain, N)."""
    _, bundle = make_test_set(chain, T_TRAIN, cfg["n_test"])
    rng = rng_for("exp10-cmp", chain.n_states, n_chains)
    A = np.stack([chain.sample(rng, N_SITES) for _ in range(n_chains)])

    t0 = time.perf_counter()
    k0 = rng.dirichlet(np.ones(chain.n_states), size=chain.n_states).T
    k_fit, trace = fit_em_discrete(
        chain.levels, k0, noisy_groups(A, T_TRAIN, rng), n_iters=300
    )
    em_seconds = time.perf_counter() - t0

    nets = {
        mode: train_dsm_denoiser(
            A, T_TRAIN, rng_for("exp10-net", chain.n_states, n_chains, mode),
            hidden=cfg["net_hidden"], n_steps=cfg["net_steps"], parameterization=mode,
        )
        for mode in PARAMETERIZATIONS
    }

    for t in T_TRAIN:
        X, m_ref = bundle[t]
        alpha, delta = alpha_delta(t)
        m_em = discrete_bp(chain.levels, k_fit, X, alpha, delta, chain.mu).means
        base = {
            "n_states": chain.n_states, "n_chains": n_chains, "t": t,
            "em_n_params": chain.n_states * (chain.n_states - 1),
            "net_n_params": nets["eps"].n_params,
            "em_seconds": em_seconds,
            "em_monotone_violation": monotone_violation(trace["log_evidence"]),
        }
        if tag_extra:
            base.update(tag_extra)
        out_rows.append({**base, "method": "em_bp",
                         **evaluate_denoiser(m_em, m_ref, X, t)})
        for mode, dsm in nets.items():
            out_rows.append({**base, "method": f"dsm_net_{mode}",
                             **evaluate_denoiser(
                                 dsm_posterior_mean(dsm, X, t), m_ref, X, t)})


def part2_headline(cfg, out):
    chain = make_random_chain(
        cfg["headline_states"], rng_for("exp10-chain", cfg["headline_states"]),
        concentration=0.6,
    )
    rows = []
    for n_chains in cfg["sizes"]:
        _compare(chain, cfg, n_chains, rows)

    fig, ax = new_figure()
    for method, style in (("em_bp", "o-"), ("dsm_net_eps", "s-"), ("dsm_net_x0", "^-")):
        agg = [
            float(np.mean([r["score_rel_l2"] for r in rows
                           if r["method"] == method and r["n_chains"] == n]))
            for n in cfg["sizes"]
        ]
        ax.loglog(cfg["sizes"], agg, style, label=method)
    ax.set_xlabel("number of training chains $N$")
    ax.set_ylabel("relative score error")
    ax.set_title(f"Discrete alphabet $S={cfg['headline_states']}$: exact reference,\n"
                 "so the margin cannot be a grid artifact")
    ax.legend()
    save_figure(fig, out / "discrete_headline.png")
    return rows


def part3_alphabet_size(cfg, out):
    rows = []
    for n_states in cfg["alphabet_sizes"]:
        chain = make_random_chain(
            n_states, rng_for("exp10-chain", n_states), concentration=0.6
        )
        _compare(chain, cfg, cfg["alphabet_n_chains"], rows)
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_10_discrete_alphabet",
        "Discrete-alphabet chains: exact BP, exact Baum-Welch, no grid error.",
    )
    args = parser.parse_args()

    quick = {
        "sizes": (64, 256), "alphabet_sizes": (3, 5), "n_rep": 2,  # frozen-exempt: compendium only, not a paper experiment
        "headline_states": 4, "alphabet_n_chains": 256,
        "net_hidden": (64, 64), "net_steps": 2000, "n_test": 128,
    }
    full = {
        "sizes": (32, 64, 128, 256, 512, 1024, 2048),
        "alphabet_sizes": (3, 4, 6, 8, 12), "n_rep": 8,
        "headline_states": 5, "alphabet_n_chains": 512,
        "net_hidden": (128, 128), "net_steps": 20000, "n_test": N_TEST,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    parts = {
        "recovery": ("Baum-Welch recovery and rate",
                     lambda o: write_csv(o / "discrete_recovery.csv",
                                         part1_recovery(cfg, o))),
        "headline": ("EM-BP vs score network, exact reference",
                     lambda o: write_csv(o / "discrete_headline.csv",
                                         part2_headline(cfg, o))),
        "alphabet": ("comparison vs alphabet size",
                     lambda o: write_csv(o / "discrete_alphabet_size.csv",
                                         part3_alphabet_size(cfg, o))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "t_train": T_TRAIN, "quick": args.quick,
        "parts": list(selected), "overrides": args.set,
        "parameterizations": PARAMETERIZATIONS, **cfg, **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
