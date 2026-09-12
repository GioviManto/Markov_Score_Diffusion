"""Experiment 15 -- which BP oracle does a learned diffusion score agree with?

This replaces the probe used in exp_13 `ordering`, which was the wrong
measurement, with the one Garnier-Brun, Mezard, Moscato and Saglietti actually
use (arXiv:2408.15138, Figs. 1c-d and 2). Their construction gives a *family*
of exact inference algorithms, BP_k for k = 0..L, one per level of hierarchical
filtering, and the question "how much of the correlation structure has the
network learned" is answered by asking which member of that family its
predictions currently match. exp_13 instead resolved the error onto hierarchy
levels and asked whether it moved -- a weaker probe, and it found nothing.

Three parts.

ladder     Analytic and free. How the speciation ladder collapses as the
           filter level rises: at level k the top k rungs merge into one, so
           the number of distinct speciation times falls from L+1 to L-k+2.
           This is the point where the two papers' devices compose.

alignment  Train a denoiser on *unfiltered* data and, at a sequence of training
           budgets, measure its distance to every BP_k. The prediction taken
           from the paper is that early training matches large k (short range)
           and later training matches decreasing k, i.e. correlations are
           acquired shortest-range first.

mismatch   Train on data filtered at k_train and ask which oracle the network
           matches when tested on unfiltered data. The paper's finding is that
           it matches BP_{k_train} -- the algorithm matched to its *training*
           distribution -- even where that is not the optimal predictor.

Reading the numbers
-------------------
`argmin_k` is only meaningful when the oracles are actually distinguishable, so
every row also carries the pairwise spread between oracles at that noise level.
Where BP_k for different k give nearly the same answer, the argmin is noise and
the CSV says so rather than leaving it to be discovered.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src import spectral
from src.denoiser import dsm_posterior_mean, train_dsm_denoiser
from src.hierarchy import GaussianTree, filtered_tree_bp_gaussian
from src.noising import alpha_delta
from src.plotting import save_figure
from src.utils import ensure_dir, rng_for, write_csv, write_json

NAME = "exp_15_bp_oracles"


def settings(quick: bool) -> dict:
    return {
        "depth": 4 if not quick else 3,
        "branching": 2,
        "rho": 0.9,
        "n_train": 1024 if not quick else 64,
        "n_test": 512 if not quick else 32,
        "t_train": (0.1, 0.2, 0.4, 0.8, 1.6),
        "hidden": (128, 128),
        "budgets": (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)
        if not quick
        else (100, 400),
        "final_steps": 32000 if not quick else 400,
        "parameterization": "eps",
        "ladder_depths": (3, 4, 5, 6) if not quick else (3, 4),
        "ladder_rhos": (0.85, 0.9),
    }


def _oracles(cfg):
    """`BP_k` for every filter level, all on the same underlying (L, rho)."""
    return {
        k: GaussianTree(depth=cfg["depth"], branching=cfg["branching"],
                        rho=cfg["rho"], filter_level=k)
        for k in range(cfg["depth"] + 1)
    }


def _oracle_means(oracles, x, t):
    alpha, delta = alpha_delta(float(t))
    return {k: filtered_tree_bp_gaussian(tree, x, alpha, delta)
            for k, tree in oracles.items()}


def _spread(means: dict) -> float:
    """Largest relative gap between any two oracles -- the resolution floor.

    If this is small the argmin over k is not measuring anything, so it travels
    with every row that reports an argmin.
    """
    keys = sorted(means)
    ref = np.linalg.norm(means[keys[0]])
    gaps = [
        np.linalg.norm(means[a] - means[b]) / max(ref, 1e-12)
        for i, a in enumerate(keys) for b in keys[i + 1:]
    ]
    return float(max(gaps)) if gaps else 0.0


# ----------------------------------------------------------------------------
# Part 1 -- how filtering collapses the ladder
# ----------------------------------------------------------------------------

def part1_ladder(cfg, out_dir):
    print("[ladder] speciation rungs versus filter level")
    rows = []
    for rho in cfg["ladder_rhos"]:
        for depth in cfg["ladder_depths"]:
            for k in range(depth + 1):
                tree = GaussianTree(depth=depth, branching=cfg["branching"],
                                    rho=rho, filter_level=k)
                spec = tree.level_eigenvalues()
                times = [float(spectral.speciation_time(lam)) for _l, lam, _m in spec]
                rows.append({
                    "rho": rho, "depth": depth, "filter_level": k,
                    "n_leaves": tree.n_leaves,
                    "n_distinct_rungs": len(spec),
                    "block_size": tree.block_size,
                    "t_speciation_top": max(times),
                    "t_speciation_bottom": min(times),
                    "ladder_span": max(times) / max(min(times), 1e-12),
                    "cross_block_covariance": tree.cross_block_covariance,
                    "levels": ";".join(str(l) for l, _x, _m in spec),
                    "eigenvalues": ";".join(f"{lam:.6f}" for _l, lam, _m in spec),
                    "speciation_times": ";".join(f"{t:.6f}" for t in times),
                })
    write_csv(out_dir / "ladder.csv", rows)

    for rho in cfg["ladder_rhos"]:
        deep = max(cfg["ladder_depths"])
        sub = [r for r in rows if r["rho"] == rho and r["depth"] == deep]
        print(f"  rho={rho}, L={deep}: rungs " +
              " -> ".join(f"k={r['filter_level']}:{r['n_distinct_rungs']}" for r in sub))
        print("    ladder span " +
              " ".join(f"{r['ladder_span']:.1f}x" for r in sub))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for rho in cfg["ladder_rhos"]:
        for depth in cfg["ladder_depths"]:
            sub = sorted([r for r in rows if r["rho"] == rho and r["depth"] == depth],
                         key=lambda r: r["filter_level"])
            axes[0].plot([r["filter_level"] for r in sub],
                         [r["n_distinct_rungs"] for r in sub], "o-",
                         label=f"L={depth}, rho={rho}")
    axes[0].set_xlabel("filter level k")
    axes[0].set_ylabel("distinct speciation times")
    axes[0].set_title("Filtering removes rungs from the ladder")
    axes[0].legend(fontsize=6)

    deep, rho = max(cfg["ladder_depths"]), cfg["ladder_rhos"][-1]
    for k in range(deep + 1):
        r = next(x for x in rows
                 if x["rho"] == rho and x["depth"] == deep and x["filter_level"] == k)
        ts = [float(s) for s in r["speciation_times"].split(";")]
        axes[1].plot([k] * len(ts), ts, "o", ms=6, alpha=0.75)
    axes[1].set_xlabel("filter level k")
    axes[1].set_ylabel(r"$t_S$ of each rung")
    axes[1].set_yscale("log")
    axes[1].set_title(f"The ladder itself (L={deep}, rho={rho})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "ladder.png")


# ----------------------------------------------------------------------------
# Part 2 -- which oracle does the network match, and when
# ----------------------------------------------------------------------------

def part2_alignment(cfg, out_dir):
    depth = cfg["depth"]
    oracles = _oracles(cfg)
    data_tree = oracles[0]                     # train on the full hierarchy
    print(f"[alignment] L={depth}, N={cfg['n_train']}, oracles k=0..{depth}")

    rng = rng_for(NAME, "train", depth, cfg["n_train"])
    a_train = data_tree.sample(rng, cfg["n_train"])
    rng_t = rng_for(NAME, "test", depth)
    a_test = data_tree.sample(rng_t, cfg["n_test"])
    x_test = {}
    for t in cfg["t_train"]:
        alpha, delta = alpha_delta(float(t))
        x_test[t] = alpha * a_test + np.sqrt(delta) * rng_t.standard_normal(a_test.shape)

    rows = []
    for steps in cfg["budgets"]:
        res = train_dsm_denoiser(
            a_train, cfg["t_train"], rng_for(NAME, "dsm", depth, cfg["n_train"]),
            hidden=tuple(cfg["hidden"]), n_steps=int(steps),
            parameterization=cfg["parameterization"],
        )
        for t in cfg["t_train"]:
            x = x_test[t]
            means = _oracle_means(oracles, x, t)
            m_net = dsm_posterior_mean(res, x, float(t))
            dist = {k: float(np.linalg.norm(m_net - m) / np.linalg.norm(m))
                    for k, m in means.items()}
            best = min(dist, key=dist.get)
            row = {"budget": int(steps), "t": t, "argmin_k": best,
                   "oracle_spread": _spread(means),
                   "best_distance": dist[best]}
            for k in range(depth + 1):
                row[f"dist_k{k}"] = dist[k]
            rows.append(row)
        summary = [r for r in rows if r["budget"] == steps]
        print(f"  {steps:>6} steps: argmin_k per t = "
              + " ".join(f"{r['t']}:{r['argmin_k']}" for r in summary)
              + f"   (spread {np.mean([r['oracle_spread'] for r in summary]):.3f})")
    write_csv(out_dir / "alignment.csv", rows)
    _plot_alignment(rows, cfg, out_dir)


def _plot_alignment(rows, cfg, out_dir):
    import matplotlib.pyplot as plt

    ts = list(cfg["t_train"])
    fig, axes = plt.subplots(1, len(ts), figsize=(3.1 * len(ts), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0, 0.9, cfg["depth"] + 1))
    for ax, t in zip(axes, ts):
        sub = sorted([r for r in rows if r["t"] == t], key=lambda r: r["budget"])
        for k in range(cfg["depth"] + 1):
            ax.loglog([r["budget"] for r in sub], [r[f"dist_k{k}"] for r in sub],
                      "o-", color=colors[k], ms=4, label=f"BP$_{k}$")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("gradient steps")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("relative distance to the oracle")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "alignment.png")


# ----------------------------------------------------------------------------
# Part 3 -- does a network trained on filtered data match its own oracle?
# ----------------------------------------------------------------------------

def part3_mismatch(cfg, out_dir):
    depth = cfg["depth"]
    oracles = _oracles(cfg)
    print("[mismatch] train on k_train, test on unfiltered data")

    rng_t = rng_for(NAME, "mismatch-test", depth)
    a_test = oracles[0].sample(rng_t, cfg["n_test"])
    x_test = {}
    for t in cfg["t_train"]:
        alpha, delta = alpha_delta(float(t))
        x_test[t] = alpha * a_test + np.sqrt(delta) * rng_t.standard_normal(a_test.shape)

    rows = []
    for k_train in range(depth + 1):
        rng = rng_for(NAME, "mismatch-train", depth, k_train)
        a_train = oracles[k_train].sample(rng, cfg["n_train"])
        res = train_dsm_denoiser(
            a_train, cfg["t_train"], rng_for(NAME, "mismatch-dsm", depth, k_train),
            hidden=tuple(cfg["hidden"]), n_steps=cfg["final_steps"],
            parameterization=cfg["parameterization"],
        )
        hits = []
        for t in cfg["t_train"]:
            x = x_test[t]
            means = _oracle_means(oracles, x, t)
            m_net = dsm_posterior_mean(res, x, float(t))
            dist = {k: float(np.linalg.norm(m_net - m) / np.linalg.norm(m))
                    for k, m in means.items()}
            best = min(dist, key=dist.get)
            hits.append(best == k_train)
            row = {"k_train": k_train, "t": t, "argmin_k": best,
                   "matches_own_oracle": int(best == k_train),
                   "oracle_spread": _spread(means)}
            for k in range(depth + 1):
                row[f"dist_k{k}"] = dist[k]
            rows.append(row)
        print(f"  k_train={k_train}: argmin_k = "
              + " ".join(f"{r['t']}:{r['argmin_k']}"
                         for r in rows if r["k_train"] == k_train)
              + f"   matched own oracle {sum(hits)}/{len(hits)}")
    write_csv(out_dir / "mismatch.csv", rows)

    n_match = sum(r["matches_own_oracle"] for r in rows)
    print(f"  overall: matched its own oracle in {n_match}/{len(rows)} cells")


PARTS = {
    "ladder": part1_ladder,
    "alignment": part2_alignment,
    "mismatch": part3_mismatch,
}


def main() -> None:
    parser = experiment_parser(NAME, __doc__)
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    cfg = apply_overrides(settings(args.quick), args.set)
    parts = select_parts(PARTS, args.only)
    out_dir = ensure_dir(args.output_dir)
    write_json(
        out_dir / f"params_{'_'.join(parts) if args.only else 'all'}.json",
        {**cfg, "quick": args.quick, "parts": list(parts),
         "overrides": args.set, **provenance()},
    )
    for fn in parts.values():
        fn(cfg, out_dir)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
