"""Experiment 14 -- collapse, memorization, and why a BP score does not collapse.

Biroli, Bonnaire, de Bortoli and Mezard (Nat. Commun. 15, 9957, 2024) identify
a *collapse* transition in the backward dynamics: past a certain time the
trajectory driven by the empirical score is captured by one training point, and
memorization can only be avoided if the dataset is exponentially large in the
dimension. That is a statement about the *empirical* score -- the score of the
measure that puts mass 1/N on each training example -- and it is the sharpest
available statement of why score-based generative modelling is hard.

The point of this experiment is that the statement does not apply to a BP
score, and that the reason is structural rather than a matter of degree.

    - The empirical score's sufficient statistic is the training set itself:
      N x n numbers, growing with both dataset size and dimension.
    - A BP score's sufficient statistic is the fitted kernel: two numbers for a
      Gaussian AR(1), or M x M for a nonparametric one, *independent of n*. The
      training data reaches it only through Xi, which is an average. There is no
      training point for a trajectory to be captured by.

So the prediction is not "BP memorizes less". It is that the memorization axis
does not exist for BP, at any n and any N, while for the empirical score it is
governed by an exactly computable excess entropy. That is a falsifiable
difference and this experiment measures it.

A closed-form collapse criterion in this setting
------------------------------------------------
For a Gaussian AR(1) chain the noised law is exactly N(0, alpha_t^2 C + Delta_t I),
so its per-site excess entropy relative to the terminal measure N(0, I) is

    s(t) = -(1 / 2n) log det(alpha_t^2 C + Delta_t I),

available in closed form at every t. Equating the entropy the dataset can supply
with the entropy the distribution demands, `log N = n s(t_C)`, gives a predicted
collapse time with no fitted constants. Part `time` tests it against the
measured concentration of the empirical score's own weights.

Parts
-----
budget    Analytic. The exponential wall: dataset size needed to avoid
          memorization, versus chain length.
collapse  Measured. Nearest-training-neighbour distance of generated samples
          for the empirical, EM-BP and DSM-network scores, swept over n and N.
          `--set include_mdn=True` adds a mixture-density BP kernel with
          hundreds of parameters, which is the answer to the obvious objection
          that a two-parameter kernel *cannot* memorize so the result is
          trivial. If the nonparametric rung also refuses, the mechanism is the
          sufficient statistic and not the parameter count.
time      Measured. Predicted vs observed collapse time for the empirical score.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src import spectral
from src.denoiser import bp_posterior_mean, dsm_posterior_mean, train_dsm_denoiser
from src.em import fit_em
from src.kernels import GaussianAR1Kernel, MDNKernel
from src.noising import alpha_delta
from src.plotting import save_figure
from src.priors import GaussianAR1
from src.reverse import reverse_sde, time_grid
from src.utils import ensure_dir, rng_for, write_csv, write_json

NAME = "exp_14_memorization_collapse"


def settings(quick: bool) -> dict:
    return {
        "rho": 0.85,
        "budget_sizes": (4, 8, 16, 24, 32, 48, 64),
        "budget_rhos": (0.5, 0.7, 0.85, 0.95),
        "n_sites": (8, 16, 32) if not quick else (8, 16),
        "train_sizes": (64, 256, 1024) if not quick else (32, 128),
        "n_gen": 128 if not quick else 32,
        "t_max": 3.0,
        "t_min": 0.02,
        "n_steps_sde": 120 if not quick else 40,
        "grid_size": 301,
        "grid_half_width": 8.0,
        "t_train": (0.1, 0.2, 0.4, 0.8, 1.6),
        "em_iters": 60 if not quick else 10,
        "hidden": (128, 128),
        "dsm_steps": 12000 if not quick else 400,
        # The nonparametric BP rung, off by default because it costs another EM
        # run per cell. Turn it on to answer "a 2-parameter kernel cannot
        # memorize, so of course it does not".
        "include_mdn": False,
        "mdn_components": 3,
        "mdn_hidden": 24,
        "mdn_iters": 40 if not quick else 8,
        # part `time`
        "time_sites": (8, 16, 32) if not quick else (8, 16),
        "time_sizes": (32, 128, 512, 2048) if not quick else (32, 128),
        "time_batch": 256 if not quick else 32,
        "time_grid_points": 80 if not quick else 25,
        "time_t_min": 1e-4,
    }


def _grid(cfg):
    grid = np.linspace(-cfg["grid_half_width"], cfg["grid_half_width"], cfg["grid_size"])
    w = np.full(grid.size, grid[1] - grid[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return grid, w


# ----------------------------------------------------------------------------
# The empirical score
# ----------------------------------------------------------------------------

def empirical_weights(x: np.ndarray, a_train: np.ndarray, alpha: float, delta: float):
    """Softmax weights of the empirical (kernel) density at `x`.

    `w[b, d]` is the posterior probability that observation `b` came from
    training chain `d`. Collapse is exactly the statement that this
    distribution concentrates on one `d`.
    """
    diff = x[:, None, :] - alpha * a_train[None, :, :]
    logits = -0.5 * np.sum(diff**2, axis=2) / delta
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def empirical_score(x: np.ndarray, a_train: np.ndarray, t: float) -> np.ndarray:
    """Score of the N-point empirical measure pushed through the OU channel.

    This is the score a model with unlimited capacity and a perfectly minimized
    training loss converges to, which is why it is the right reference for
    memorization rather than a strawman.
    """
    alpha, delta = alpha_delta(t)
    w = empirical_weights(x, a_train, alpha, delta)
    return (alpha * (w @ a_train) - x) / delta


# ----------------------------------------------------------------------------
# Part 1 -- the exponential wall
# ----------------------------------------------------------------------------

def part1_budget(cfg, out_dir):
    print("[budget] dataset size required to avoid memorization")
    rows = []
    for rho in cfg["budget_rhos"]:
        rate = spectral.gaussian_chain_excess_entropy_rate(rho)
        for n in cfg["budget_sizes"]:
            rows.append({
                "rho": rho, "n_sites": n,
                "excess_entropy_rate_nats": rate,
                "excess_entropy_total_nats": spectral.gaussian_chain_excess_entropy(n, rho),
                "collapse_dataset_size": spectral.collapse_dataset_size(n, rho),
                "log10_collapse_dataset_size": float(
                    spectral.gaussian_chain_excess_entropy(n, rho) / np.log(10)
                ),
            })
    write_csv(out_dir / "budget.csv", rows)
    for rho in cfg["budget_rhos"]:
        r = [x for x in rows if x["rho"] == rho]
        print(f"  rho={rho}: {r[0]['excess_entropy_rate_nats']:.4f} nats/site; "
              f"n={r[-1]['n_sites']} needs 1e{r[-1]['log10_collapse_dataset_size']:.1f} chains")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rho in cfg["budget_rhos"]:
        r = [x for x in rows if x["rho"] == rho]
        ax.semilogy([x["n_sites"] for x in r],
                    [x["collapse_dataset_size"] for x in r], "o-", label=f"rho={rho}")
    ax.axhline(1e6, color="k", ls="--", lw=0.8)
    ax.text(cfg["budget_sizes"][0], 1.4e6, "a large dataset", fontsize=8)
    ax.set_xlabel("chain length n")
    ax.set_ylabel("dataset size needed to avoid memorization")
    ax.set_title("The curse of dimensionality for a memorizing score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save_figure(fig, out_dir / "budget.png")


# ----------------------------------------------------------------------------
# Part 2 -- measured memorization of three scores
# ----------------------------------------------------------------------------

def _nearest_distance(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-sample distance to the nearest row of `reference`, per coordinate."""
    n = samples.shape[1]
    d2 = (
        np.sum(samples**2, axis=1)[:, None]
        - 2.0 * samples @ reference.T
        + np.sum(reference**2, axis=1)[None, :]
    )
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0) / n)


def part2_collapse(cfg, out_dir):
    grid, w = _grid(cfg)
    rho = cfg["rho"]
    prior = GaussianAR1(rho=rho)
    rows = []

    for n in cfg["n_sites"]:
        for n_train in cfg["train_sizes"]:
            print(f"[collapse] n={n}, N={n_train} "
                  f"(wall at N ~ {spectral.collapse_dataset_size(n, rho):.3g})")
            rng = rng_for(NAME, "data", n, n_train)
            a_train = np.stack([prior.sample(rng, n) for _ in range(n_train)])

            groups = []
            for t in cfg["t_train"]:
                alpha, delta = alpha_delta(float(t))
                x = alpha * a_train + np.sqrt(delta) * rng.standard_normal(a_train.shape)
                groups.append((x, alpha, delta))
            fitted, trace = fit_em(
                GaussianAR1Kernel(rho=0.3, q=0.8), grid, w, groups,
                n_iters=cfg["em_iters"],
            )
            net = train_dsm_denoiser(
                a_train, cfg["t_train"], rng_for(NAME, "dsm", n, n_train),
                hidden=tuple(cfg["hidden"]), n_steps=cfg["dsm_steps"],
                parameterization="eps",
            )

            # The obvious objection to "BP does not memorize" is that a
            # two-parameter kernel *cannot* memorize, so the result is trivial.
            # The MDN rung answers it: a mixture-density kernel with hundreds of
            # parameters, fitted by the same EM, still has a sufficient statistic
            # that is an average over edges and still does not grow with n. If it
            # also refuses to memorize, the mechanism is the statistic and not
            # the parameter count.
            mdn = None
            if cfg["include_mdn"]:
                mdn, mdn_trace = fit_em(
                    MDNKernel.init(cfg["mdn_components"], cfg["mdn_hidden"],
                                   rng_for(NAME, "mdn-init", n, n_train)),
                    grid, w, groups, n_iters=cfg["mdn_iters"],
                )
                print(f"    MDN kernel: {mdn.net.n_params} params, "
                      f"monotone violation {mdn_trace.monotone_violation:.2e}")

            scores = {
                "empirical": lambda x, t: empirical_score(x, a_train, float(t)),
                "bp_em": lambda x, t: -(
                    x - np.exp(-float(t))
                    * bp_posterior_mean(fitted, grid, w, x, float(t))
                ) / (1.0 - np.exp(-2.0 * float(t))),
                "bp_true": lambda x, t: -(
                    x - np.exp(-float(t))
                    * bp_posterior_mean(prior, grid, w, x, float(t))
                ) / (1.0 - np.exp(-2.0 * float(t))),
                "dsm_eps": lambda x, t: -(
                    x - np.exp(-float(t)) * dsm_posterior_mean(net, x, float(t))
                ) / (1.0 - np.exp(-2.0 * float(t))),
            }
            if mdn is not None:
                scores["bp_em_mdn"] = lambda x, t: -(
                    x - np.exp(-float(t))
                    * bp_posterior_mean(mdn, grid, w, x, float(t))
                ) / (1.0 - np.exp(-2.0 * float(t)))

            times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps_sde"])
            # A held-out true sample is the no-memorization yardstick: whatever
            # nearest-neighbour distance a genuinely fresh draw has, a
            # non-memorizing generator should match it.
            fresh = np.stack(
                [prior.sample(rng_for(NAME, "fresh", n, n_train, i), n)
                 for i in range(cfg["n_gen"])]
            )
            base = float(np.mean(_nearest_distance(fresh, a_train)))

            for name, fn in scores.items():
                gen_rng = rng_for(NAME, "gen", n, n_train)
                x_init = gen_rng.standard_normal((cfg["n_gen"], n))
                out = reverse_sde(x_init, fn, times, gen_rng)
                nn = float(np.mean(_nearest_distance(out, a_train)))
                rows.append({
                    "n_sites": n, "n_train": n_train, "method": name,
                    "nn_distance": nn,
                    "nn_distance_fresh": base,
                    "memorization_ratio": nn / max(base, 1e-12),
                    "sample_std": float(np.std(out)),
                    "lag1_corr": float(
                        np.mean([np.corrcoef(out[:, i], out[:, i + 1])[0, 1]
                                 for i in range(n - 1)])
                    ),
                    "collapse_wall": spectral.collapse_dataset_size(n, rho),
                    "rho_hat": float(fitted.rho) if name == "bp_em" else "",
                    "em_monotone_violation": (
                        float(trace.monotone_violation) if name == "bp_em" else ""
                    ),
                })
                print(f"    {name:<10} nn/fresh = {nn / base:.3f}  "
                      f"(nn {nn:.4f}, fresh {base:.4f})  "
                      f"std {rows[-1]['sample_std']:.3f}  "
                      f"lag1 {rows[-1]['lag1_corr']:.3f}")
    write_csv(out_dir / "collapse.csv", rows)
    _plot_collapse(rows, cfg, out_dir)


def _plot_collapse(rows, cfg, out_dir):
    import matplotlib.pyplot as plt

    methods = ["empirical", "dsm_eps", "bp_em", "bp_em_mdn", "bp_true"]
    fig, axes = plt.subplots(1, len(cfg["n_sites"]),
                             figsize=(3.4 * len(cfg["n_sites"]), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, cfg["n_sites"]):
        for m in methods:
            sub = sorted([r for r in rows if r["n_sites"] == n and r["method"] == m],
                         key=lambda r: r["n_train"])
            if sub:
                ax.semilogx([r["n_train"] for r in sub],
                            [r["memorization_ratio"] for r in sub], "o-", label=m)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("training chains N")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nearest-train distance / fresh-sample distance")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "collapse.png")


# ----------------------------------------------------------------------------
# Part 3 -- predicted vs measured collapse time
# ----------------------------------------------------------------------------

def _excess_entropy_at_t(n: int, rho: float, t: float) -> float:
    """Per-site excess entropy of the noised law, relative to N(0, I)."""
    alpha, delta = alpha_delta(t)
    cov = alpha**2 * spectral.chain_covariance(n, rho) + delta * np.eye(n)
    _sign, logdet = np.linalg.slogdet(cov)
    return float(-logdet / (2.0 * n))


def _predicted_collapse_time(n: int, n_train: int, rho: float,
                             times: np.ndarray) -> float:
    """Solve n * s(t) = log N for t, by interpolation on `times`.

    `n s(t)` is the entropy the noised distribution demands and `log N` the
    entropy the dataset can supply; the crossing is where the dataset stops
    being able to cover it. NaN when the crossing lies outside the window --
    in particular when `log N` already exceeds the t = 0 excess entropy, which
    is the regime where the dataset is *above* the wall and never collapses.
    """
    s = np.array([n * _excess_entropy_at_t(n, rho, float(t)) for t in times])
    return _threshold_crossing(times, s, np.log(n_train))


def _threshold_crossing(times, curve, level):
    """Time at which `curve` crosses `level`, interpolated in log t.

    Direction-agnostic on purpose: the weight entropy and the participation
    ratio *increase* with t, while the entropy demand `n s(t)` *decreases* with
    it, and both are compared against a level here. Locating the sign change
    rather than the first index below the level keeps one function correct for
    both. Returns NaN when the level is not crossed inside the window, so that
    a window edge is never reported as a measurement.
    """
    order = np.argsort(np.asarray(times, dtype=float))
    tt = np.asarray(times, dtype=float)[order]
    dd = np.asarray(curve, dtype=float)[order] - level
    sign_change = np.where(dd[:-1] * dd[1:] < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])
    x0, x1, y0, y1 = np.log(tt[i]), np.log(tt[i + 1]), dd[i], dd[i + 1]
    return float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))


def part3_time(cfg, out_dir):
    rho = cfg["rho"]
    prior = GaussianAR1(rho=rho)
    # The window has to reach well below the reverse sampler's t_min: collapse
    # of the *weights* happens later (smaller t) than collapse of the samples,
    # and a crossing outside the window reads as "no collapse".
    times = np.geomspace(cfg["t_max"], cfg["time_t_min"], cfg["time_grid_points"])
    rows = []
    print("[time] predicted vs measured collapse time for the empirical score")

    for n in cfg["time_sites"]:
        for n_train in cfg["time_sizes"]:
            rng = rng_for(NAME, "time-data", n, n_train)
            a_train = np.stack([prior.sample(rng, n) for _ in range(n_train)])
            pred = _predicted_collapse_time(n, n_train, rho, times)

            # Two probes, because they answer different questions. Noising a
            # *training* chain follows what an empirical-score trajectory
            # actually does, and reproduces the classical collapse. Noising a
            # *fresh* chain asks whether the training set can explain a typical
            # sample at all -- the generalization side of the same coin.
            probes = {
                "train": a_train[: cfg["time_batch"]],
                "fresh": np.stack(
                    [prior.sample(rng_for(NAME, "time-test", n, i), n)
                     for i in range(cfg["time_batch"])]
                ),
            }
            row = {
                "n_sites": n, "n_train": n_train,
                "t_collapse_predicted": pred,
                "log_n_train": float(np.log(n_train)),
                "excess_entropy_total_at_0": spectral.gaussian_chain_excess_entropy(n, rho),
            }
            for probe, a_probe in probes.items():
                part_curve, ent_curve = [], []
                for t in times:
                    alpha, delta = alpha_delta(float(t))
                    x = alpha * a_probe + np.sqrt(delta) * rng.standard_normal(
                        a_probe.shape
                    )
                    w = empirical_weights(x, a_train, alpha, delta)
                    # Participation ratio: effective number of training points
                    # in play, N when none dominates and 1 when one does.
                    part_curve.append(float(np.mean(1.0 / np.sum(w**2, axis=1))))
                    ent_curve.append(float(
                        np.mean(-np.sum(w * np.log(np.maximum(w, 1e-300)), axis=1))
                    ))
                # Midpoint of the entropy collapse: the weight entropy falls
                # from log N to 0, so half-way is the scale-free landmark, and
                # it is the one the entropic criterion is about. The
                # participation threshold is reported alongside as a check that
                # the answer does not hinge on the choice of landmark.
                row[f"t_collapse_{probe}_entropy"] = _threshold_crossing(
                    times, ent_curve, 0.5 * np.log(n_train)
                )
                row[f"t_collapse_{probe}_part2"] = _threshold_crossing(
                    times, part_curve, 2.0
                )
                row[f"participation_min_{probe}"] = float(part_curve[-1])
                row[f"participation_max_{probe}"] = float(part_curve[0])
                row[f"curve_participation_{probe}"] = ";".join(
                    f"{c:.4f}" for c in part_curve
                )
                row[f"curve_entropy_{probe}"] = ";".join(f"{c:.4f}" for c in ent_curve)
            rows.append(row)
            print(f"  n={n:>3} N={n_train:>5}: predicted {pred:.4f}  "
                  f"measured(train) {row['t_collapse_train_entropy']:.4f}  "
                  f"measured(fresh) {row['t_collapse_fresh_entropy']:.4f}  "
                  f"participation {row['participation_max_train']:.1f} -> "
                  f"{row['participation_min_train']:.2f}")

    write_csv(out_dir / "collapse_time.csv", rows)
    write_csv(
        out_dir / "collapse_time_summary.csv",
        [{k: v for k, v in r.items() if not k.startswith("curve")} for r in rows],
    )

    for probe in ("train", "fresh"):
        ok = [(r["t_collapse_predicted"], r[f"t_collapse_{probe}_entropy"])
              for r in rows
              if np.isfinite(r[f"t_collapse_{probe}_entropy"])
              and np.isfinite(r["t_collapse_predicted"])]
        if len(ok) >= 3:
            p, m = np.array(ok).T
            if np.std(p) > 0 and np.std(m) > 0:
                print(f"  [{probe}] predicted vs measured: Pearson r = "
                      f"{np.corrcoef(p, m)[0, 1]:+.3f} over {len(ok)} settings; "
                      f"ratio measured/predicted "
                      f"{np.mean(m / p):.2f} +/- {np.std(m / p):.2f}")
            else:
                print(f"  [{probe}] not enough spread to correlate "
                      f"({len(ok)} settings)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    pts = []
    for n in cfg["time_sites"]:
        sub = [r for r in rows if r["n_sites"] == n]
        xs = [r["t_collapse_predicted"] for r in sub]
        ys = [r["t_collapse_train_entropy"] for r in sub]
        pts += [(a, b) for a, b in zip(xs, ys) if np.isfinite(a) and np.isfinite(b)]
        ax.loglog(xs, ys, "o", label=f"n={n}")
    if pts:
        lo = min(min(p) for p in pts) * 0.5
        hi = max(max(p) for p in pts) * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("predicted collapse time  (n s(t) = log N)")
    ax.set_ylabel("measured collapse time  (weight entropy = ½ log N)")
    ax.set_title("Collapse time from the excess entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "collapse_time.png")


# ----------------------------------------------------------------------------

PARTS = {
    "budget": part1_budget,
    "collapse": part2_collapse,
    "time": part3_time,
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
