"""Experiment 13 -- hierarchical priors: the speciation ladder and what learns it.

This experiment sits at the intersection of two papers the project was pointed
at. Note that the intersection is *not* unoccupied: Sclocchi, Favero and Wyart
(PNAS 122(1) e2408799121, 2025) study diffusion on hierarchical data and find a
sharp transition for high-level features against a smooth evolution of
low-level ones. What the Gaussian tree adds is that the whole ladder of
crossover times is analytic, so the measurements below are calibration against
a known answer rather than discovery. See docs/PAPER_CONNECTIONS.md.

Biroli, Bonnaire, de Bortoli and Mezard (Nat. Commun. 15, 9957, 2024) show that
the backward diffusion passes through a *speciation* cross-over set by the top
eigenvalue of the data covariance, and that in high dimension this time diverges
logarithmically because that eigenvalue grows with the dimension.

Garnier-Brun, Mezard, Moscato and Saglietti (arXiv:2408.15138) show that on
hierarchically generated sequences a transformer reconstructs correlations
*sequentially*, shortest length scale first, and that the reference against
which this is measured is exact BP inference.

Put together they give something specific and checkable here: a hierarchical
prior has a *ladder* of covariance eigenvalues, one per level, so the reverse
diffusion should show not one speciation time but L + 1 of them, resolving the
hierarchy coarse-to-fine, at times computable in closed form.

Parts
-----
spectra   Analytic. Chain vs tree spectra and their speciation times. Establishes
          that a stationary chain has a *bounded* top eigenvalue however long it
          is -- no diverging speciation time, no cascade -- while a tree's grows
          geometrically in depth, i.e. logarithmically in dimension, which is
          the regime the Nature Communications analysis describes.
cascade   Measured. Per-level commitment along both the forward process and the
          exact-score reverse SDE, against the predicted ladder.
levels    Which method learns which level: per-level posterior-mean error for
          exact BP, EM-fitted BP, and a DSM network, at matched data budget.
ordering  Level-resolved error *as a function of training step*: the diffusion
          analogue of the sequential-inclusion result.
block_independent
          Sweep the correlation range by making blocks independent. NOT the
          paper's filtering -- see exp_15_bp_oracles.py for that.
speciation Symmetry breaking, not just an information cross-over. A two-component
          root prior keeps the covariance -- and therefore every predicted time
          -- exactly unchanged while making the coarsest transition a genuine
          choice between classes, measured through the exact class posterior
          P(root > 0 | x_t) that BP returns.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src import spectral
from src.denoiser import dsm_posterior_mean, train_dsm_denoiser
from src.hierarchy import (
    GaussianTree,
    fit_em_tree,
    tree_bp_gaussian,
    tree_bp_grid,
    tree_root_belief,
)
from src.kernels import GaussianAR1Kernel
from src.noising import alpha_delta
from src.plotting import save_figure
from src.reverse import reverse_sde, time_grid
from src.utils import ensure_dir, rng_for, write_csv, write_json

NAME = "exp_13_speciation_cascade"


def settings(quick: bool) -> dict:
    return {
        "chain_sizes": (8, 16, 32, 64, 128, 256, 512),
        "chain_rhos": (0.5, 0.85, 0.95),
        "tree_depths": (2, 3, 4, 5, 6) if not quick else (2, 3, 4),
        "tree_rhos": (0.7, 0.85, 0.9),
        "branching": 2,
        # cascade: the exact-score dynamics is cheap, so it runs deeper than
        # the learning parts, which pay for tree EM at every iteration.
        "cascade_depth": 5 if not quick else 3,
        "rho": 0.9,
        "n_samples": 4000 if not quick else 400,
        "n_paths": 512 if not quick else 64,
        "t_max": 3.0,
        "t_min": 0.02,
        "n_steps_sde": 200 if not quick else 40,
        # learning
        "depth": 4 if not quick else 3,
        "grid_size": 301,
        "grid_half_width": 8.0,
        "n_train": 256 if not quick else 64,
        "n_test": 256 if not quick else 32,
        "t_train": (0.1, 0.2, 0.4, 0.8, 1.6),
        "em_iters": 150 if not quick else 20,
        "hidden": (128, 128),
        "dsm_steps": 20000 if not quick else 500,
        "probe_steps": (250, 500, 1000, 2000, 4000, 8000, 16000)
        if not quick
        else (100, 250, 500),
        # part `speciation`: grid BP on every reverse step, so smaller than the
        # Gaussian cascade, which has a closed-form message update.
        "spec_depth": 4 if not quick else 2,
        "root_separation": 0.9,
        "spec_paths": 256 if not quick else 32,
        "spec_steps": 120 if not quick else 30,
    }


def _grid(cfg):
    grid = np.linspace(-cfg["grid_half_width"], cfg["grid_half_width"], cfg["grid_size"])
    w = np.full(grid.size, grid[1] - grid[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return grid, w


# ----------------------------------------------------------------------------
# Part 1 -- spectra and the predicted ladder
# ----------------------------------------------------------------------------

def part1_spectra(cfg, out_dir):
    print("[spectra] chain and tree covariance spectra ...")

    chain_rows = []
    for rho in cfg["chain_rhos"]:
        limit = spectral.chain_top_eigenvalue_limit(rho)
        for n in cfg["chain_sizes"]:
            top = float(spectral.chain_spectrum(n, rho).max())
            chain_rows.append({
                "family": "chain", "rho": rho, "dimension": n,
                "top_eigenvalue": top,
                "top_eigenvalue_limit": limit,
                "fraction_of_limit": top / limit,
                "t_speciation": float(spectral.speciation_time(top)),
                "t_speciation_limit": float(spectral.speciation_time(limit)),
            })
    write_csv(out_dir / "spectra_chain.csv", chain_rows)

    tree_rows = []
    for rho in cfg["tree_rhos"]:
        for depth in cfg["tree_depths"]:
            tree = GaussianTree(depth=depth, branching=cfg["branching"], rho=rho)
            spec = sorted(tree.level_eigenvalues(), key=lambda r: r[0])
            top = spec[0][1]
            for level, lam, mult in spec:
                tree_rows.append({
                    "family": "tree", "rho": rho, "depth": depth,
                    "dimension": tree.n_leaves,
                    "level": level, "eigenvalue": lam, "multiplicity": mult,
                    "t_speciation": float(spectral.speciation_time(lam)),
                    "top_eigenvalue": top,
                    "t_speciation_top": float(spectral.speciation_time(top)),
                    "block_size": 1 if level < 0 else cfg["branching"] ** (depth - level),
                })
    write_csv(out_dir / "spectra_tree.csv", tree_rows)

    for rho in cfg["tree_rhos"]:
        deep = max(cfg["tree_depths"])
        tree = GaussianTree(depth=deep, branching=cfg["branching"], rho=rho)
        tops = [
            GaussianTree(depth=d, branching=cfg["branching"], rho=rho).subtree_row_sum(0)
            for d in cfg["tree_depths"]
        ]
        growth = tops[-1] / tops[0]
        print(
            f"  tree rho={rho}: top eigenvalue {tops[0]:.2f} -> {tops[-1]:.2f} "
            f"over depths {cfg['tree_depths'][0]}..{deep} ({growth:.1f}x); "
            f"t_S {spectral.speciation_time(tops[0]):.3f} -> "
            f"{spectral.speciation_time(tops[-1]):.3f}; "
            f"levels span t_S in "
            f"[{min(spectral.speciation_time(l) for _, l, _ in tree.level_eigenvalues()):.3f},"
            f" {max(spectral.speciation_time(l) for _, l, _ in tree.level_eigenvalues()):.3f}]"
        )
    for rho in cfg["chain_rhos"]:
        lim = spectral.chain_top_eigenvalue_limit(rho)
        small = spectral.chain_spectrum(cfg["chain_sizes"][0], rho).max()
        big = spectral.chain_spectrum(cfg["chain_sizes"][-1], rho).max()
        print(
            f"  chain rho={rho}: top eigenvalue {small:.2f} -> {big:.2f} "
            f"(limit {lim:.2f}); t_S saturates at "
            f"{spectral.speciation_time(lim):.3f}"
        )

    _plot_spectra(cfg, chain_rows, tree_rows, out_dir)


def _plot_spectra(cfg, chain_rows, tree_rows, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for rho in cfg["chain_rhos"]:
        rows = [r for r in chain_rows if r["rho"] == rho]
        ax.semilogx([r["dimension"] for r in rows],
                    [r["t_speciation"] for r in rows],
                    "o-", label=f"chain rho={rho}")
    for rho in cfg["tree_rhos"]:
        rows = sorted({(r["dimension"], r["t_speciation_top"])
                       for r in tree_rows if r["rho"] == rho})
        ax.semilogx([d for d, _ in rows], [t for _, t in rows],
                    "s--", label=f"tree rho={rho}")
    ax.set_xlabel("dimension")
    ax.set_ylabel(r"$t_S$ of the top mode")
    ax.set_title("Speciation time: chain saturates, tree diverges")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    deep = max(cfg["tree_depths"])
    for rho in cfg["tree_rhos"]:
        rows = sorted(
            [r for r in tree_rows if r["rho"] == rho and r["depth"] == deep],
            key=lambda r: r["level"],
        )
        ax.plot([r["level"] for r in rows], [r["t_speciation"] for r in rows],
                "o-", label=f"rho={rho}")
    ax.set_xlabel("hierarchy level (-1 = whole tree, larger = finer)")
    ax.set_ylabel(r"$t_S$")
    ax.set_title(f"The ladder at depth {deep}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "spectra.png")


# ----------------------------------------------------------------------------
# Part 2 -- the measured cascade
# ----------------------------------------------------------------------------

def _crossing_time(times, values, level=1.0 / np.sqrt(2.0)):
    """Linear interpolation of where a decreasing curve crosses `level`."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(t)
    t, v = t[order], v[order]
    below = np.where(v < level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    x0, x1, y0, y1 = t[i - 1], t[i], v[i - 1], v[i]
    if y1 == y0:
        return float(x0)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def part2_cascade(cfg, out_dir):
    depth, rho, b = cfg["cascade_depth"], cfg["rho"], cfg["branching"]
    tree = GaussianTree(depth=depth, branching=b, rho=rho)
    v, levels = tree.level_projector_basis()
    lam_of_level = {lev: lam for lev, lam, _ in tree.level_eigenvalues()}
    uniq = sorted(lam_of_level)
    print(f"[cascade] tree L={depth} b={b} rho={rho}, {tree.n_leaves} leaves, "
          f"{len(uniq)} levels")

    times = np.geomspace(cfg["t_max"], cfg["t_min"], 40)

    # -- forward process: the analytic statement, checked by sampling ------
    rng = rng_for(NAME, "cascade-forward", depth, rho)
    a = tree.sample(rng, cfg["n_samples"])
    proj0 = a @ v
    forward = {lev: [] for lev in uniq}
    for t in times:
        alpha, delta = alpha_delta(float(t))
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
        projt = x @ v
        for lev in uniq:
            cols = np.where(levels == lev)[0]
            c = [np.corrcoef(proj0[:, k], projt[:, k])[0, 1] for k in cols]
            forward[lev].append(float(np.mean(c)))

    # -- reverse SDE with the exact score ----------------------------------
    rng_rev = rng_for(NAME, "cascade-reverse", depth, rho)
    sde_times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps_sde"])
    x_init = rng_rev.standard_normal((cfg["n_paths"], tree.n_leaves))

    recorded: list[tuple[float, np.ndarray]] = []

    def score_fn(x, t):
        alpha, delta = alpha_delta(float(t))
        m = tree_bp_gaussian(tree, x, alpha, delta)
        return -(x - alpha * m) / delta

    def cb(t, x, _s):
        recorded.append((float(t), (x @ v).copy()))

    x_final = reverse_sde(x_init, score_fn, sde_times, rng_rev, callback=cb)
    proj_final = x_final @ v

    rows = []
    for lev in uniq:
        cols = np.where(levels == lev)[0]
        pred = float(spectral.speciation_time(lam_of_level[lev]))
        fwd_curve = forward[lev]
        rev_t, rev_curve = [], []
        for t, projt in recorded:
            c = [np.corrcoef(proj_final[:, k], projt[:, k])[0, 1] for k in cols]
            rev_t.append(t)
            rev_curve.append(float(np.mean(c)))
        rows.append({
            "level": lev,
            "eigenvalue": lam_of_level[lev],
            "multiplicity": int((levels == lev).sum()),
            "t_speciation_predicted": pred,
            "t_crossing_forward": _crossing_time(times, fwd_curve),
            "t_crossing_reverse": _crossing_time(rev_t, rev_curve),
            "commitment_at_prediction": float(
                np.interp(pred, times[::-1], np.asarray(fwd_curve)[::-1])
            ),
        })
        rows[-1]["curve_forward"] = ";".join(f"{c:.5f}" for c in fwd_curve)
        rows[-1]["curve_reverse"] = ";".join(f"{c:.5f}" for c in rev_curve)
    write_csv(out_dir / "cascade.csv", rows)
    write_csv(
        out_dir / "cascade_times.csv",
        [{k: v for k, v in r.items() if not k.startswith("curve")} for r in rows],
    )

    for r in rows:
        print(f"  level {r['level']:>2}: Lambda={r['eigenvalue']:7.3f}  "
              f"t_S pred {r['t_speciation_predicted']:.3f}  "
              f"forward {r['t_crossing_forward']:.3f}  "
              f"reverse {r['t_crossing_reverse']:.3f}")

    _plot_cascade(times, forward, recorded, proj_final, levels, uniq,
                  lam_of_level, out_dir)


def _plot_cascade(times, forward, recorded, proj_final, levels, uniq,
                  lam_of_level, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(uniq)))
    for c, lev in zip(colors, uniq):
        axes[0].semilogx(times, forward[lev], "o-", color=c, ms=3,
                         label=f"level {lev}")
        axes[0].axvline(spectral.speciation_time(lam_of_level[lev]),
                        color=c, ls=":", alpha=0.7)
        cols = np.where(levels == lev)[0]
        rev_t = [t for t, _ in recorded]
        rev_c = [float(np.mean([np.corrcoef(proj_final[:, k], p[:, k])[0, 1]
                                for k in cols])) for _, p in recorded]
        axes[1].semilogx(rev_t, rev_c, "-", color=c, label=f"level {lev}")
        axes[1].axvline(spectral.speciation_time(lam_of_level[lev]),
                        color=c, ls=":", alpha=0.7)
    for ax, title in zip(axes, ["forward process", "reverse SDE (exact score)"]):
        ax.axhline(1 / np.sqrt(2), color="k", ls="--", lw=0.8)
        ax.set_xlabel("t")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("commitment (correlation with the final sample)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "cascade.png")


# ----------------------------------------------------------------------------
# Learning on a tree: shared setup for parts 3 and 4
# ----------------------------------------------------------------------------

def _make_training_data(cfg, tree):
    rng = rng_for(NAME, "train", cfg["depth"], cfg["rho"], cfg["n_train"])
    a_train = tree.sample(rng, cfg["n_train"])
    groups = []
    for t in cfg["t_train"]:
        alpha, delta = alpha_delta(float(t))
        x = alpha * a_train + np.sqrt(delta) * rng.standard_normal(a_train.shape)
        groups.append((x, alpha, delta))
    return a_train, groups, rng


def _make_test_data(cfg, tree):
    rng = rng_for(NAME, "test", cfg["depth"], cfg["rho"])
    a = tree.sample(rng, cfg["n_test"])
    per_t = {}
    for t in cfg["t_train"]:
        alpha, delta = alpha_delta(float(t))
        per_t[t] = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
    return a, per_t


def _level_errors(err, m_ref, v, levels, uniq):
    """Posterior-mean error resolved by hierarchy level, three ways.

    Three, because one is not enough to say anything honest. The *relative*
    error divides by the reference magnitude in that subspace, which shrinks
    with the level's eigenvalue -- so fine levels look worse for every method,
    including a method with uniform absolute error, and the ranking across
    levels is then an artefact of the normalization rather than a finding. The
    *absolute* per-mode RMS has no such bias, and the *share* of total squared
    error is normalization-free and answers the question the hierarchy actually
    poses: where does a given method's error live?
    """
    pe = err @ v
    pr = m_ref @ v
    total_sq = float(np.sum(pe**2))
    rel, absolute, share = {}, {}, {}
    for lev in uniq:
        cols = np.where(levels == lev)[0]
        num = float(np.sqrt(np.mean(pe[:, cols] ** 2)))
        den = float(np.sqrt(np.mean(pr[:, cols] ** 2)))
        rel[lev] = num / max(den, 1e-12)
        absolute[lev] = num
        share[lev] = float(np.sum(pe[:, cols] ** 2)) / max(total_sq, 1e-300)
    return rel, absolute, share


def _level_columns(row, uniq, rel, absolute, share):
    for lev in uniq:
        row[f"level_{lev}"] = rel[lev]
        row[f"abs_level_{lev}"] = absolute[lev]
        row[f"share_level_{lev}"] = share[lev]
    return row


# ----------------------------------------------------------------------------
# Part 3 -- which method learns which level
# ----------------------------------------------------------------------------

def part3_levels(cfg, out_dir):
    depth, rho, b = cfg["depth"], cfg["rho"], cfg["branching"]
    tree = GaussianTree(depth=depth, branching=b, rho=rho)
    v, levels = tree.level_projector_basis()
    uniq = sorted({int(x) for x in levels})
    grid, w = _grid(cfg)
    print(f"[levels] N={cfg['n_train']} trees, {tree.n_leaves} leaves, "
          f"{len(uniq)} levels")

    a_train, groups, rng = _make_training_data(cfg, tree)
    _a_test, x_test = _make_test_data(cfg, tree)

    fitted, trace = fit_em_tree(
        GaussianAR1Kernel(rho=0.3, q=0.8), grid, w, groups, b, depth,
        n_iters=cfg["em_iters"],
    )
    print(f"  EM: rho_hat={fitted.rho:.4f} (true {rho}), q_hat={fitted.q:.4f} "
          f"(true {1 - rho**2:.4f}), {len(trace.log_evidence)} iters, "
          f"monotone violation {trace.monotone_violation:.2e}")
    log_k_hat = fitted.log_transition_matrix(grid)
    log_root = -0.5 * grid**2 - 0.5 * np.log(2 * np.pi)

    nets = {}
    for par in ("eps", "x0"):
        nets[par] = train_dsm_denoiser(
            a_train, cfg["t_train"],
            rng_for(NAME, "dsm", par, cfg["n_train"]),
            hidden=tuple(cfg["hidden"]), n_steps=cfg["dsm_steps"],
            parameterization=par,
        )
        print(f"  DSM/{par}: {nets[par].n_params} params, "
              f"{nets[par].seconds:.1f}s")

    rows = []
    for t in cfg["t_train"]:
        alpha, delta = alpha_delta(float(t))
        x = x_test[t]
        m_ref = tree_bp_gaussian(tree, x, alpha, delta)

        estimates = {
            "bp_exact_grid": tree_bp_grid(
                GaussianAR1Kernel(rho=rho, q=1 - rho**2).log_transition_matrix(grid),
                grid, log_root, x, alpha, delta, b, depth),
            "bp_em": tree_bp_grid(log_k_hat, grid, log_root, x, alpha, delta, b, depth),
        }
        for par in ("eps", "x0"):
            estimates[f"dsm_{par}"] = dsm_posterior_mean(nets[par], x, float(t))

        for method, m_hat in estimates.items():
            rel, absolute, share = _level_errors(m_hat - m_ref, m_ref, v, levels, uniq)
            row = {
                "t": t, "method": method,
                "n_train": cfg["n_train"],
                "rel_error_total": float(
                    np.linalg.norm(m_hat - m_ref) / np.linalg.norm(m_ref)
                ),
            }
            rows.append(_level_columns(row, uniq, rel, absolute, share))
    write_csv(out_dir / "levels.csv", rows)

    for t in cfg["t_train"]:
        print(f"  t={t}")
        for r in [r for r in rows if r["t"] == t]:
            per = " ".join(f"{r[f'abs_level_{lev}']:.4f}" for lev in uniq)
            shr = " ".join(f"{r[f'share_level_{lev}']:.2f}" for lev in uniq)
            print(f"    {r['method']:<14} total {r['rel_error_total']:.4f}   "
                  f"abs[{','.join(str(l) for l in uniq)}] {per}   share {shr}")

    _plot_levels(rows, uniq, cfg, out_dir)


def _plot_levels(rows, uniq, cfg, out_dir):
    import matplotlib.pyplot as plt

    ts = list(cfg["t_train"])
    fig, axes = plt.subplots(1, len(ts), figsize=(3.1 * len(ts), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, ts):
        for r in [r for r in rows if r["t"] == t]:
            ax.semilogy(uniq, [max(r[f"abs_level_{lev}"], 1e-16) for lev in uniq],
                        "o-", label=r["method"], ms=4)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("level (coarse -> fine)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("absolute posterior-mean error (per mode, RMS)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "levels.png")


# ----------------------------------------------------------------------------
# Part 4 -- the order in which levels are acquired during training
# ----------------------------------------------------------------------------

def part4_ordering(cfg, out_dir):
    depth, rho, b = cfg["depth"], cfg["rho"], cfg["branching"]
    tree = GaussianTree(depth=depth, branching=b, rho=rho)
    v, levels = tree.level_projector_basis()
    uniq = sorted({int(x) for x in levels})
    grid, w = _grid(cfg)
    t_probe = 0.4
    alpha, delta = alpha_delta(t_probe)
    print(f"[ordering] level-resolved error vs training budget, t={t_probe}")

    a_train, groups, _rng = _make_training_data(cfg, tree)
    _a_test, x_test = _make_test_data(cfg, tree)
    x = x_test[min(cfg["t_train"], key=lambda s: abs(s - t_probe))]
    t_probe = min(cfg["t_train"], key=lambda s: abs(s - t_probe))
    alpha, delta = alpha_delta(float(t_probe))
    m_ref = tree_bp_gaussian(tree, x, alpha, delta)
    log_root = -0.5 * grid**2 - 0.5 * np.log(2 * np.pi)

    rows = []

    # DSM: independent runs at increasing step budgets. Restarting from the same
    # seed at each budget keeps the trajectory identical, so the sequence is a
    # snapshot of one run rather than a set of unrelated ones.
    for steps in cfg["probe_steps"]:
        res = train_dsm_denoiser(
            a_train, cfg["t_train"], rng_for(NAME, "dsm-order", cfg["n_train"]),
            hidden=tuple(cfg["hidden"]), n_steps=int(steps), parameterization="eps",
        )
        m_hat = dsm_posterior_mean(res, x, float(t_probe))
        rel, absolute, share = _level_errors(m_hat - m_ref, m_ref, v, levels, uniq)
        row = {"method": "dsm_eps", "budget": int(steps), "budget_unit": "grad_steps",
               "rel_error_total": float(
                   np.linalg.norm(m_hat - m_ref) / np.linalg.norm(m_ref)),
               "rho_hat": "", "monotone_violation": ""}
        rows.append(_level_columns(row, uniq, rel, absolute, share))
        print(f"  dsm {steps:>6} steps: total {row['rel_error_total']:.4f}  "
              + " ".join(f"L{lev}={absolute[lev]:.4f}" for lev in uniq))

    # EM: the same probe against the number of EM iterations. Run *incrementally*
    # -- each budget continues the previous fit rather than restarting -- so the
    # sequence is one trajectory sampled at increasing budgets, and costs
    # max(budget) iterations instead of their sum.
    fitted = GaussianAR1Kernel(rho=0.3, q=0.8)
    em_budgets = [1, 2, 4, 8, 16, 32, 64, 128, cfg["em_iters"]]
    em_budgets = sorted({min(bgt, cfg["em_iters"]) for bgt in em_budgets})
    done = 0
    worst_violation = 0.0
    for iters in em_budgets:
        fitted, trace = fit_em_tree(
            fitted, grid, w, groups, b, depth, n_iters=int(iters) - done, tol=0.0
        )
        done = int(iters)
        worst_violation = max(worst_violation, trace.monotone_violation)
        m_hat = tree_bp_grid(
            fitted.log_transition_matrix(grid), grid, log_root, x, alpha, delta,
            b, depth,
        )
        rel, absolute, share = _level_errors(m_hat - m_ref, m_ref, v, levels, uniq)
        row = {"method": "bp_em", "budget": int(iters), "budget_unit": "em_iters",
               "rel_error_total": float(
                   np.linalg.norm(m_hat - m_ref) / np.linalg.norm(m_ref)),
               "rho_hat": float(fitted.rho),
               "monotone_violation": float(worst_violation)}
        rows.append(_level_columns(row, uniq, rel, absolute, share))
        print(f"  em  {iters:>6} iters: rho={fitted.rho:.4f} "
              f"total {row['rel_error_total']:.4f}  "
              + " ".join(f"L{lev}={absolute[lev]:.4f}" for lev in uniq))

    write_csv(out_dir / "ordering.csv", rows)
    _plot_ordering(rows, uniq, out_dir)


def _plot_ordering(rows, uniq, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, method, xlabel in zip(
        axes, ("dsm_eps", "bp_em"), ("gradient steps", "EM iterations")
    ):
        sub = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: r["budget"])
        if not sub:
            continue
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(uniq)))
        for c, lev in zip(colors, uniq):
            ax.loglog([r["budget"] for r in sub], [r[f"abs_level_{lev}"] for r in sub],
                      "o-", color=c, ms=4, label=f"level {lev}")
        ax.set_xlabel(xlabel)
        ax.set_title(method)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("absolute posterior-mean error (per mode, RMS)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "ordering.png")


# ----------------------------------------------------------------------------

def part5_block_independent(cfg, out_dir):
    """Correlation range swept by making blocks INDEPENDENT.

    **This is not the hierarchical filtering of arXiv:2408.15138, and an earlier
    version of this docstring said it was.** That paper draws the depth-k nodes
    conditionally independently *given the root*, so the blocks stay correlated
    through it; here the blocks are made outright independent, which puts a zero
    where the paper's construction puts rho^{2L}. The correct construction now
    lives in `GaussianTree(filter_level=k)` and the experiment built on it is
    `exp_15_bp_oracles.py`, which is also where the paper's actual probe --
    comparison against the family of mismatched BP_k oracles -- is implemented.

    What remains here is still a well-defined and separately interesting sweep:
    block-independent truncation, i.e. the harsher cut in which nothing survives
    between blocks at all. The sequence length, the training budget and the
    network are held fixed; only the correlation range moves.

    The prediction being tested is the diffusion analogue of P1's sequential
    acquisition: at fixed budget the network's deficit should grow with k,
    because longer-range structure is what it acquires last, while EM-BP's
    should not, because the object it estimates -- one transition kernel -- is
    the same size whatever k is.

    The asymmetry is deliberate and is the premise of the whole package, so it
    is worth restating: BP is *given* the graph and learns only the kernel on
    it, exactly as BP is the model-aware reference in P1. The network is given
    the sequence. This measures how the difficulty of the learning problem
    scales with correlation range for each, not who wins a fair fight.
    """
    b, rho = cfg["branching"], cfg["rho"]
    depth = cfg["depth"]
    n_leaves = b**depth
    grid, w = _grid(cfg)
    log_root = -0.5 * grid**2 - 0.5 * np.log(2 * np.pi)
    print(f"[block_independent] {n_leaves} leaves, block level k = 1..{depth}")

    rows = []
    for k in range(1, depth + 1):
        block = b**k
        n_blocks = n_leaves // block
        sub = GaussianTree(depth=k, branching=b, rho=rho)

        rng = rng_for(NAME, "filter-train", k, cfg["n_train"])
        a_train = sub.sample(rng, cfg["n_train"] * n_blocks).reshape(
            cfg["n_train"], n_leaves
        )
        groups = []
        for t in cfg["t_train"]:
            alpha, delta = alpha_delta(float(t))
            x = alpha * a_train + np.sqrt(delta) * rng.standard_normal(a_train.shape)
            groups.append((x.reshape(-1, block), alpha, delta))

        rng_t = rng_for(NAME, "filter-test", k)
        a_test = sub.sample(rng_t, cfg["n_test"] * n_blocks).reshape(
            cfg["n_test"], n_leaves
        )

        fitted, trace = fit_em_tree(
            GaussianAR1Kernel(rho=0.3, q=0.8), grid, w, groups, b, k,
            n_iters=cfg["em_iters"],
        )
        net = train_dsm_denoiser(
            a_train, cfg["t_train"], rng_for(NAME, "filter-dsm", k),
            hidden=tuple(cfg["hidden"]), n_steps=cfg["dsm_steps"],
            parameterization="eps",
        )

        for t in cfg["t_train"]:
            alpha, delta = alpha_delta(float(t))
            x = alpha * a_test + np.sqrt(delta) * rng_t.standard_normal(a_test.shape)
            flat = x.reshape(-1, block)
            m_ref = tree_bp_gaussian(sub, flat, alpha, delta).reshape(x.shape)
            m_em = tree_bp_grid(
                fitted.log_transition_matrix(grid), grid, log_root, flat,
                alpha, delta, b, k,
            ).reshape(x.shape)
            m_net = dsm_posterior_mean(net, x, float(t))

            denom = np.linalg.norm(m_ref)
            # Absolute RMS as well as relative: ||m_ref|| itself changes with k
            # (weaker correlation means less shrinkage structure to recover), so
            # a relative error compared *across* k is partly reading its own
            # denominator. The ratio `advantage` is immune, sharing a reference.
            rms = float(np.sqrt(m_ref.size))
            rows.append({
                "filter_level": k, "block_size": block, "n_blocks": n_blocks,
                "t": t, "n_train": cfg["n_train"],
                "rel_error_bp_em": float(np.linalg.norm(m_em - m_ref) / denom),
                "rel_error_dsm": float(np.linalg.norm(m_net - m_ref) / denom),
                "abs_error_bp_em": float(np.linalg.norm(m_em - m_ref) / rms),
                "abs_error_dsm": float(np.linalg.norm(m_net - m_ref) / rms),
                "reference_rms": float(denom / rms),
                "advantage": float(
                    np.linalg.norm(m_net - m_ref) / max(np.linalg.norm(m_em - m_ref), 1e-12)
                ),
                "top_eigenvalue": sub.level_eigenvalues()[0][1],
                "t_speciation_top": float(
                    spectral.speciation_time(sub.level_eigenvalues()[0][1])
                ),
                "rho_hat": float(fitted.rho),
                "em_monotone_violation": float(trace.monotone_violation),
            })
        mid = [r for r in rows if r["filter_level"] == k]
        print(f"  k={k} (blocks of {block:>2}): rho_hat={fitted.rho:.4f}  "
              f"abs err  EM {np.mean([r['abs_error_bp_em'] for r in mid]):.4f}  "
              f"DSM {np.mean([r['abs_error_dsm'] for r in mid]):.4f}  "
              f"(ref rms {np.mean([r['reference_rms'] for r in mid]):.3f})  "
              f"advantage {np.mean([r['advantage'] for r in mid]):.2f}x")

    write_csv(out_dir / "block_independent.csv", rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ks = sorted({r["filter_level"] for r in rows})
    for t in cfg["t_train"]:
        sub_rows = [r for r in rows if r["t"] == t]
        axes[0].plot(ks, [r["rel_error_dsm"] for r in sub_rows], "o-", label=f"DSM t={t}")
        axes[0].plot(ks, [r["rel_error_bp_em"] for r in sub_rows], "s--",
                     label=f"EM-BP t={t}")
        axes[1].plot(ks, [r["advantage"] for r in sub_rows], "o-", label=f"t={t}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("filter level k (correlation range)")
    axes[0].set_ylabel("relative posterior-mean error")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_xlabel("filter level k (correlation range)")
    axes[1].set_ylabel("DSM error / EM-BP error")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "block_independent.png")


def part6_speciation(cfg, out_dir):
    """Genuine symmetry breaking, at the time the Gaussian cross-over predicts.

    Everything else in this experiment uses a Gaussian prior, where there is no
    class to speciate *into*: `commitment` locates an information cross-over,
    which shares P2's criterion but is not P2's phenomenon. Giving the root a
    symmetric two-component prior fixes that. Because the mixture keeps unit
    variance, **the leaf covariance and therefore every eigenvalue and every
    predicted time are exactly unchanged** -- only the modality moves. So this
    is a controlled test of a sharp claim: the same `t_S` that locates the
    information cross-over should locate the class choice.

    The order parameter is the **class posterior**, `P(root > 0 | x_t)`, which
    BP returns exactly (`tree_root_belief`) at every noise level. Using the sign
    of the tree-mean instead would conflate the class with the within-class
    fluctuation, since the uniform mode carries both; the root posterior does
    not. What is measured is how strongly the class inferred at time `t`
    already agrees with the class of the finished sample, and the magnetization
    `|2P − 1|` alongside it.
    """
    b, rho = cfg["branching"], cfg["rho"]
    depth = cfg["spec_depth"]
    mu = cfg["root_separation"]
    grid, w = _grid(cfg)

    gaussian = GaussianTree(depth=depth, branching=b, rho=rho)
    bimodal = GaussianTree(depth=depth, branching=b, rho=rho, root_separation=mu)
    assert np.allclose(gaussian.leaf_covariance(), bimodal.leaf_covariance())

    v, levels = bimodal.level_projector_basis()
    uniform = v[:, 0]
    lam_top = bimodal.level_eigenvalues()[0][1]
    t_pred = float(spectral.speciation_time(lam_top))
    print(f"[speciation] depth {depth}, mu={mu}, Lambda_top={lam_top:.3f}, "
          f"t_S predicted {t_pred:.3f}")

    log_k = GaussianAR1Kernel(rho=rho, q=1 - rho**2).log_transition_matrix(grid)
    times = time_grid(cfg["t_max"], cfg["t_min"], cfg["spec_steps"])

    pos = grid > 0.0

    def class_posterior(x, t, log_root):
        alpha, delta = alpha_delta(float(t))
        belief = tree_root_belief(
            log_k, grid, log_root, x, alpha, delta, b, depth
        )
        return belief[:, pos].sum(axis=1)

    rows = []
    for name, tree in (("bimodal", bimodal), ("gaussian", gaussian)):
        log_root = tree.log_root_density(grid)

        def score_fn(x, t, _lr=log_root):
            alpha, delta = alpha_delta(float(t))
            m = tree_bp_grid(log_k, grid, _lr, x, alpha, delta, b, depth)
            return -(x - alpha * m) / delta

        rng = rng_for(NAME, "speciation", name, depth, mu)
        x_init = rng.standard_normal((cfg["spec_paths"], tree.n_leaves))
        recorded: list[tuple[float, np.ndarray, np.ndarray]] = []

        def cb(t, x, _s, _lr=log_root):
            recorded.append((float(t), class_posterior(x, t, _lr), (x @ uniform).copy()))

        x_final = reverse_sde(x_init, score_fn, times, rng, callback=cb)
        final_p = class_posterior(x_final, float(times[-1]), log_root)
        final_class = final_p > 0.5
        final_proj = x_final @ uniform

        agree, magnet, corr, ts = [], [], [], []
        for t, p, proj in recorded:
            ts.append(t)
            agree.append(float(np.mean((p > 0.5) == final_class)))
            magnet.append(float(np.mean(np.abs(2.0 * p - 1.0))))
            corr.append(float(np.corrcoef(proj, final_proj)[0, 1]))

        # Class agreement runs from 1/2 (undecided) to 1 (decided), so 3/4 is
        # the "half the information is in" landmark, matching what 1/sqrt(2)
        # marks for the correlation.
        t_class = _crossing_time(ts, agree, level=0.75)
        t_corr = _crossing_time(ts, corr, level=1.0 / np.sqrt(2.0))
        rows.append({
            "root": name, "depth": depth,
            "root_separation": mu if name == "bimodal" else 0.0,
            "top_eigenvalue": lam_top,
            "t_speciation_predicted": t_pred,
            "t_class_agreement_075": t_class,
            "t_correlation_crossing": t_corr,
            "magnetization_final": magnet[-1],
            "magnetization_initial": magnet[0],
            "n_paths": cfg["spec_paths"],
            # Binomial standard error on the agreement curve: the crossing can
            # only be located to about this much, and at small path counts that
            # is the dominant uncertainty, not the integrator.
            "agreement_stderr": float(0.5 / np.sqrt(cfg["spec_paths"])),
            "curve_times": ";".join(f"{t:.5f}" for t in ts),
            "curve_class_agreement": ";".join(f"{a:.5f}" for a in agree),
            "curve_magnetization": ";".join(f"{a:.5f}" for a in magnet),
            "curve_correlation": ";".join(f"{c:.5f}" for c in corr),
        })
        print(f"  {name:<9} class-agreement 0.75 at {t_class:.3f}, "
              f"correlation crossing at {t_corr:.3f}  (predicted {t_pred:.3f});  "
              f"magnetization {magnet[0]:.3f} -> {magnet[-1]:.3f}")

    write_csv(out_dir / "speciation.csv", rows)
    write_csv(
        out_dir / "speciation_summary.csv",
        [{k: val for k, val in r.items() if not k.startswith("curve")} for r in rows],
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in rows:
        ts = [float(s) for s in r["curve_times"].split(";")]
        ag = [float(s) for s in r["curve_class_agreement"].split(";")]
        ax.semilogx(ts, ag, "-", label=f"{r['root']} root")
    ax.axvline(t_pred, color="k", ls=":", lw=1.0, label=r"predicted $t_S$")
    ax.axhline(0.75, color="k", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("P(class at t = class of the finished sample)")
    ax.set_title("Class choice locks in at the predicted speciation time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "speciation.png")


PARTS = {
    "spectra": part1_spectra,
    "cascade": part2_cascade,
    "levels": part3_levels,
    "ordering": part4_ordering,
    "block_independent": part5_block_independent,
    "speciation": part6_speciation,
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
