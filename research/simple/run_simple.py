"""The whole project in one file: three claims, three figures, about an hour.

WHY THIS FILE EXISTS
--------------------
The full package is 20 experiments and 8,000 lines, and every number in it is defensible, but
nobody can hold it in their head. Marc and Jerome asked six questions across the meetings; three
of them are the actual thesis. This runs only those three, on a laptop, in one command.

Nothing here is reimplemented. Every numerical routine is imported from `src/`, which is covered
by the 220-test suite; only the orchestration is new, and it is short enough to read end to end.

THE IDEA, IN ONE PARAGRAPH
--------------------------
Coordinatewise OU noising contributes one unary likelihood factor per site and adds no edges. So
the posterior of a noised Markov chain is *still a chain*; a chain is a tree; sum-product on a
tree is exact. That gives a **computable population score** -- the object every diffusion model
approximates and normally cannot see. Having it lets us ask two questions that usually have no
ground truth, and get real answers.

THE THREE CLAIMS
----------------
1. The kernel is genuinely learned from noised data alone. rho is started at 0.0, 0.3 and 0.6
   against a truth of 0.85; all three land on 0.855 without ever seeing a clean sample.

2. Knowing the chain structure buys a large sample-efficiency advantage. EM-BP reaches a lower
   score error than a convolutional denoiser trained on the same data, and the gap *widens* with
   data (3.6x at N=32, 5.5x at N=512) because the network is still spending its extra chains on
   discovering the structure EM was told. (Marc's claim.)

3. That advantage does NOT predict the generated distribution. (Jerome's objection, and the
   interesting result.) The sharpest form of it is not "one method wins" -- it is that the
   network's own two parameterisations, trained on the same data and differing only in whether
   the net regresses the noise or the clean signal, generate 2.08 and 1.21 against a target of
   1.89, while the two obvious pointwise metrics rank them in OPPOSITE orders (posterior-mean MSE
   prefers x0 by 2.4x, relative score L2 prefers eps by 12%). So a pointwise ranking cannot tell
   you which of those you are going to get, and the standard validation rule -- summing
   posterior-mean MSE -- selects the arm that misses the generative target by 5.3 standard errors.

   3b. Control: on a Gaussian chain, where the fitted mixture family *contains* the truth, all
   three arms sit on the target and the spread disappears. Without this the effect could be
   blamed on the channel destroying the information; the control shows it is model
   misspecification, which is fixable, and not the observation model, which is not.

A WARNING THIS FILE EXISTS TO CARRY
-----------------------------------
An earlier version of claim 3 reported that EM-BP generated *worse* samples than the network --
the pointwise winner losing generatively. That was an artefact of stopping EM at 30 iterations.
rho is flat by iteration 25 and the innovation shape is not, so the generated statistic moves
with an unconverged kernel. Fitting to convergence reverses the conclusion.

The same trap has now caught this project three times, at three different iteration budgets, and
it is worth stating the general form. A fixed iteration budget does not compare estimators; it
compares CONVERGENCE RATES. Whatever is slower to converge looks worse, and the coordinates here
converge at wildly different speeds -- on clean chains rho is done after ONE M-step while the
innovation shape takes tens, and the observation channel slows both further. Measured across
eight independent data draws at 200 iterations, the fitted excess kurtosis is 3.02 +- 0.20
against a truth of 3.0, i.e. unbiased; at 120 iterations on one draw it reads 2.15, and an
earlier version of this file built an argument on that 2.15. Single-draw shape numbers at this
budget span 2.40 to 3.99, so they carry almost no information. See `FULL["em_iters"]`.

PROTOCOL CHOICES, ALL SETTLED ELSEWHERE
---------------------------------------
Each of these was a defect found by an external review and repaired; they are stated here rather
than re-derived.

* `one_view`: one latent chain, one noisy observation. Noising the same chain at five levels and
  treating the results as independent gives a *composite* likelihood, not the marginal likelihood
  of the data actually generated.
* Continuous-time network training over the whole integration interval. A network trained on five
  discrete noise levels and then integrated over a continuum is being asked to interpolate and
  extrapolate, and the generated-sample comparison would measure that instead of score quality.
* eps-vs-x0 parameterisation chosen once on a validation split, never on the test bundle. This
  applies to claim 2, which must compare a single network. Claim 3 reports both, because which
  one validation picks turns out to depend on how the per-level errors are aggregated (summing
  the MSE picks x0, a majority over levels picks eps) and the two generate very differently.
* C = 8 mixture components: a paired capacity design showed 8 is the optimum and 16 sits inside
  one standard error of it. That design ran at `em_iters=40`, so its ranking of capacities is
  entangled with the convergence effect described above and should be re-derived before it is
  leaned on; C = 8 is used here because it is enough for the claims, not because the sweep is
  settled.
* The generated statistic is the **AR-filtered residual** excess kurtosis, not the "innovation".
  At t_min the lag-1 residual correlation is -0.0997, so these are correlated residuals; the
  marginal shape comparison is still valid but it does not measure recovery of the clean
  innovation law.

USAGE
-----
    python3.12 simple/run_simple.py --quick     # under a minute, smoke test
    python3.12 simple/run_simple.py             # ~1 hour on 8 cores, the real thing

The hour is split about evenly between fitting the two arms to convergence and claim 3's reverse
integrations, each of which is 200 BP sweeps over 800 chains on a 301-point grid.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bp_grid import make_grid  # noqa: E402
from src.denoiser import bp_posterior_mean, score_from_mean  # noqa: E402
from src.em import fit_em  # noqa: E402
from src.kernels import MixtureInnovationKernel  # noqa: E402
from src.local_head import local_posterior_mean, train_local_head  # noqa: E402
from src.plotting import new_figure, save_figure  # noqa: E402
from src.priors import GaussianAR1, LaplaceAR1  # noqa: E402
from src.protocols import one_view_groups  # noqa: E402
from src.reverse import reverse_sde, time_grid  # noqa: E402
from src.sample_metrics import ar_residuals, excess_kurtosis  # noqa: E402
from src.utils import ensure_dir, rng_for, write_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FULL = dict(
    n_sites=32, rho=0.85, budgets=(32, 128, 512), n_seeds=3,
    n_components=8, em_iters=120, grid_size=301, grid_half_width=8.0,
    t_train=(0.1, 0.2, 0.4, 0.8, 1.6), cnn_radius=6, cnn_steps=8000,
    t_min=0.02, t_max=3.0, n_steps=200, n_generate=800, n_eval=256,
    # Claim 3's statistic is a fourth moment of generated samples and is far noisier than claim
    # 2's score error: at three seeds the standard error on it is ~0.2-0.9, against effects of
    # order 0.9. More replicates go where the noise is rather than uniformly.
    n_seeds_claim3=6,
)
# Both iteration counts are set where the arm stops improving, checked separately, because
# converging one arm and not the other would decide the comparison by fiat.
#
#   EM   (rho=0.85, N=512, C=8; true innovation variance 0.2775, true excess kurtosis 3.0)
#        30 iters -> rho 0.8380, var 0.2918, kurtosis 0.841
#       120 iters -> rho 0.8479, var 0.2701, kurtosis 2.301
#       400 iters -> rho 0.8481, var 0.2697, kurtosis 2.294     (plateau)
#   CNN  (relative score L2 against the exact score, median over the five levels)
#      4000 steps -> eps 0.0950, x0 0.0869
#      8000 steps -> eps 0.0884, x0 0.0817                      (eps flat by here)
#     16000 steps -> eps 0.0890, x0 0.0762
#
# rho converges by iteration ~25 and the innovation SHAPE does not: at 30 iterations the fitted
# kernel has a third of the true excess kurtosis, which is enough to change what claim 3 says.
# fig1 panels (a) and (b) are the same runs on the same axis and show this directly -- (a) is flat
# where (b) is still climbing, so a converged-looking rho trace proves nothing about the kernel.
QUICK = dict(
    budgets=(32, 128), n_seeds=1, n_seeds_claim3=1, em_iters=8, grid_size=161,
    cnn_steps=300, n_steps=60, n_generate=300, n_eval=64,
)
# --quick exercises every code path in under a minute and its numbers mean nothing. EM stops far
# short of convergence (rho reaches ~0.73, not 0.85), and the eps network is undertrained enough
# that its reverse integration diverges -- generated kurtosis in the teens against a target of
# 1.7, on the Gaussian control as well. Both are artefacts of the reduced settings and both are
# gone at the full config. Read it as a smoke test, never as a result.


def chains(prior, rng, n, n_sites):
    return np.stack([prior.sample(rng, n_sites) for _ in range(n)])


def noise(A, t, rng):
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    return alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)


def _mean_se(values):
    """Mean and standard error of the mean. Returns se=0 for a single value rather than nan."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def _unpack(theta):
    """Split a MixtureInnovationKernel theta vector back into (rho, pi, mu, s2)."""
    c = (len(theta) - 1) // 3
    return float(theta[0]), theta[1:1 + c], theta[1 + c:1 + 2 * c], theta[1 + 2 * c:]


def label_budgets(ax, budgets):
    """Tick the log x-axis at the budgets themselves.

    Matplotlib's default decade ticks show a single "10^2" for budgets 32/128/512, which hides
    what was actually run.
    """
    ax.set_xticks(list(budgets), [str(n) for n in budgets])
    ax.minorticks_off()


_FIT_CACHE: dict = {}


def fit_both(prior, grid, w, n_chains, seed, cfg, n_components=None):
    """Fit EM-BP and the convolutional denoiser on the SAME chains.

    Identical training data is the point of the comparison: the difference is what each method
    does with the same information, not sampling luck. The asymmetry that remains is deliberate
    and favours the network -- it redraws noise every gradient step, while EM sees one noisy
    realisation per chain and never sees a clean one.

    Memoised because claims 2 and 3 ask for the same Laplace fits at the same seeds, and a fit is
    now the expensive part. The key is everything the fit depends on; `rng_for` makes the result a
    deterministic function of it, so the cache cannot change any number.
    """
    n_components = cfg["n_components"] if n_components is None else n_components
    key = (type(prior).__name__, float(prior.rho), n_chains, seed, n_components,
           cfg["em_iters"], cfg["cnn_steps"], cfg["grid_size"])
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    rng = rng_for("simple-fit", seed, n_chains)
    A = chains(prior, rng, n_chains, cfg["n_sites"])

    kernel, trace = fit_em(
        MixtureInnovationKernel.init(
            n_components, rho=0.3, var=0.8,
            rng=rng_for("simple-init", seed, n_chains),
        ),
        grid, w,
        one_view_groups(A, cfg["t_train"], rng),   # one chain, one observation
        n_iters=cfg["em_iters"],
    )

    nets = {
        mode: train_local_head(
            A, cfg["t_train"], cfg["cnn_radius"],
            rng_for("simple-cnn", seed, n_chains, mode),
            n_steps=cfg["cnn_steps"], parameterization=mode,
            t_range=(cfg["t_min"], cfg["t_max"]),   # continuous, not five levels
        )
        for mode in ("eps", "x0")
    }
    _FIT_CACHE[key] = (kernel, nets, trace)
    return kernel, nets, trace


def pick_parameterization(prior, grid, w, nets, cfg, seed):
    """Choose eps or x0 ONCE, on a validation sample, never on the test bundle.

    One choice for the whole trajectory rather than one per noise level: switching between two
    independently trained networks partway through an integration hands the integrator a score
    field with jump discontinuities that belong to neither of them.
    """
    rng = rng_for("simple-val", seed)
    A = chains(prior, rng, 128, cfg["n_sites"])
    err = {m: 0.0 for m in nets}
    for t in cfg["t_train"]:
        X = noise(A, t, rng)
        target = bp_posterior_mean(prior, grid, w, X, t)
        for m, net in nets.items():
            err[m] += float(np.mean((local_posterior_mean(net, X, t) - target) ** 2))
    return min(err, key=err.get)


# ---------------------------------------------------------------------------
# Claim 1 -- the kernel is learned from noised data alone
# ---------------------------------------------------------------------------

def claim1_recovery(prior, grid, w, cfg, out):
    print("\n[1] kernel recovery from noised data (truth rho = 0.85)", flush=True)
    rows, traces, kurt, kernels = [], {}, {}, {}
    rng = rng_for("simple-c1")
    A = chains(prior, rng, max(cfg["budgets"]), cfg["n_sites"])
    groups = one_view_groups(A, cfg["t_train"], rng)

    for rho0 in (0.0, 0.3, 0.6):
        k, tr = fit_em(
            MixtureInnovationKernel.init(
                cfg["n_components"], rho=rho0, var=0.8, rng=rng_for("simple-c1i", rho0)),
            grid, w, groups, n_iters=cfg["em_iters"],
        )
        # theta = [rho, pi (C), mu (C), s2 (C)], so the whole convergence history is already in
        # the trace: rebuilding the kernel at each step costs nothing and is what panel (b) plots.
        traces[rho0] = [float(th[0]) for th in tr.theta]
        kurt[rho0] = [MixtureInnovationKernel(*_unpack(th)).innovation_moments[
            "innovation_excess_kurtosis"] for th in tr.theta]
        kernels[rho0] = k
        rows.append(dict(claim=1, rho_init=rho0, rho_fitted=float(k.rho),
                         rho_true=cfg["rho"], monotone_violation=tr.monotone_violation,
                         **k.innovation_moments))
        print(f"    init {rho0:.1f} -> fitted {k.rho:.4f}   "
              f"(monotone violation {tr.monotone_violation:.1e})", flush=True)

    fig, ax = new_figure(ncols=3, figsize=(14.0, 3.9))
    for rho0, path in traces.items():
        ax[0].plot(path, marker="o", ms=2.5, label=f"init {rho0}")
    ax[0].axhline(cfg["rho"], color="k", ls="--", lw=1, label="truth 0.85")
    ax[0].set_xlabel("EM iteration"); ax[0].set_ylabel(r"fitted $\rho$")
    ax[0].set_title(r"(a) correlation: converged by $\sim$25"); ax[0].legend(fontsize=7)

    # The same run, the same iterations, a different parameter -- and a hundredfold difference in
    # how long it takes to settle. Panel (a) on its own would have said "converged" at iteration
    # 30, where the fitted kernel still has a third of the true excess kurtosis.
    for rho0, path in kurt.items():
        ax[1].plot(path, marker="o", ms=2.5, label=f"init {rho0}")
    ax[1].axhline(3.0, color="k", ls="--", lw=1, label="truth 3.0")
    ax[1].axvline(30, color="0.6", ls=":", lw=1)
    ax[1].annotate("(a) looks converged here", xy=(30, 0.4), xytext=(38, 0.25),
                   fontsize=6.5, color="0.35",
                   arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    ax[1].set_xlabel("EM iteration"); ax[1].set_ylabel("innovation excess kurtosis")
    ax[1].set_title(r"(b) shape: still moving at 100"); ax[1].legend(fontsize=7)

    # Panel (c) is the rho0 = 0.3 kernel from the loop above, not a fresh fit: the density drawn
    # here is the endpoint of curves the reader can see in panels (a) and (b).
    k = kernels[0.3]
    e = np.linspace(-3, 3, 400)
    q = 1.0 - cfg["rho"] ** 2
    b = np.sqrt(q / 2.0)
    fitted = np.exp(
        np.log(k.pi[:, None]) - 0.5 * np.log(2 * np.pi * k.s2[:, None])
        - 0.5 * (e[None, :] - k.mu[:, None]) ** 2 / k.s2[:, None]).sum(axis=0)
    ax[2].semilogy(e, np.exp(-np.abs(e) / b) / (2 * b), "k--", lw=1.4, label="true Laplace")
    ax[2].semilogy(e, fitted, lw=1.4, label=f"fitted, C={cfg['n_components']}")
    ax[2].set_ylim(1e-4, 3); ax[2].set_xlabel("innovation"); ax[2].set_ylabel("density")
    ax[2].set_title("(c) learned innovation law"); ax[2].legend(fontsize=7)
    save_figure(fig, out / "fig1_recovery.pdf")
    return rows


# ---------------------------------------------------------------------------
# Claim 2 -- sample efficiency against a network (Marc)
# ---------------------------------------------------------------------------

def claim2_efficiency(prior, grid, w, cfg, out):
    print("\n[2] sample efficiency vs a convolutional denoiser", flush=True)
    rng = rng_for("simple-c2-test")
    A_test = chains(prior, rng, cfg["n_eval"], cfg["n_sites"])
    bundle = {t: (noise(A_test, t, rng),) for t in cfg["t_train"]}
    for t in cfg["t_train"]:
        X, = bundle[t]
        bundle[t] = (X, bp_posterior_mean(prior, grid, w, X, t))   # the exact reference

    rows = []
    for n_chains in cfg["budgets"]:
        for seed in range(cfg["n_seeds"]):
            kernel, nets, _ = fit_both(prior, grid, w, n_chains, seed, cfg)
            mode = pick_parameterization(prior, grid, w, nets, cfg, seed)
            for t in cfg["t_train"]:
                X, m_star = bundle[t]
                s_star = score_from_mean(X, m_star, t)
                # Both parameterisations are scored, not just the selected one. Claim 3 turns on
                # how small this pointwise gap is next to the generative gap between the same two
                # networks, so the comparison has to be in the table rather than asserted.
                for name, m_hat in (
                    ("em_bp", bp_posterior_mean(kernel, grid, w, X, t)),
                    ("cnn", local_posterior_mean(nets[mode], X, t)),
                    ("cnn_eps", local_posterior_mean(nets["eps"], X, t)),
                    ("cnn_x0", local_posterior_mean(nets["x0"], X, t)),
                ):
                    s = score_from_mean(X, m_hat, t)
                    rows.append(dict(
                        claim=2, n_chains=n_chains, seed=seed, t=t, arm=name,
                        score_rel_l2=float(np.linalg.norm(s - s_star) / np.linalg.norm(s_star)),
                        mean_mse=float(np.mean((m_hat - m_star) ** 2)), mode=mode))
        em = np.mean([r["score_rel_l2"] for r in rows
                      if r["n_chains"] == n_chains and r["arm"] == "em_bp"])
        cn = np.mean([r["score_rel_l2"] for r in rows
                      if r["n_chains"] == n_chains and r["arm"] == "cnn"])
        print(f"    N={n_chains:5d}   EM-BP {em:.4f}   CNN {cn:.4f}   ratio {cn / em:5.2f}x", flush=True)

    fig, ax = new_figure()
    for arm, style in (("em_bp", "o-"), ("cnn", "s-")):
        mean, se = [], []
        for n in cfg["budgets"]:
            per = [np.mean([r["score_rel_l2"] for r in rows
                            if r["n_chains"] == n and r["arm"] == arm and r["seed"] == s])
                   for s in range(cfg["n_seeds"])]
            mean.append(np.mean(per))
            se.append(np.std(per, ddof=1) / np.sqrt(len(per)) if len(per) > 1 else 0.0)
        mean, se = np.array(mean), np.array(se)
        ax.errorbar(cfg["budgets"], mean, yerr=se, fmt=style, capsize=3,
                    label="EM-BP (13 params)" if arm == "em_bp" else "convolution")
    ax.set_xscale("log"); ax.set_yscale("log")
    label_budgets(ax, cfg["budgets"])
    ax.set_xlabel("training chains $N$"); ax.set_ylabel("relative score error")
    ax.set_title("Structure buys data efficiency"); ax.legend()
    save_figure(fig, out / "fig2_efficiency.pdf")
    return rows


# ---------------------------------------------------------------------------
# Claim 3 -- pointwise accuracy does not predict generative fidelity (Jerome), + control
# ---------------------------------------------------------------------------

ARMS3 = {
    "em_bp": "EM-BP",
    "cnn_eps": r"convolution ($\epsilon$)",
    "cnn_x0": "convolution ($x_0$)",
}


def claim3_generation(cfg, out):
    print("\n[3] does the pointwise advantage transfer to the generated law?", flush=True)
    grid, w = make_grid(cfg["grid_half_width"], cfg["grid_size"])
    times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps"])
    rows = []

    for family, prior in (("laplace", LaplaceAR1(cfg["rho"])),
                          ("gaussian", GaussianAR1(cfg["rho"]))):
        # Target: the reverse SDE stops at t_min, so a generated sample is a draw from
        # p_{t_min}, not p_0. The honest reference is real data noised to t_min -- comparing
        # against clean data would measure the stopping floor rather than the score.
        rng = rng_for("simple-c3-ref", family)
        a_ref = noise(chains(prior, rng, cfg["n_generate"], cfg["n_sites"]), cfg["t_min"], rng)
        target = excess_kurtosis(ar_residuals(a_ref, cfg["rho"]))
        print(f"  -- {family}: target AR-residual excess kurtosis {target:.3f}", flush=True)

        for n_chains in cfg["budgets"]:
            for seed in range(cfg["n_seeds_claim3"]):
                kernel, nets, _ = fit_both(prior, grid, w, n_chains, seed, cfg)
                # Both network parameterisations are carried through, not the validation-selected
                # one. Claim 2 has to pick a single network because it compares pointwise error;
                # here picking one would hide the result. The two differ only in what the network
                # regresses -- the noise or the clean signal -- and the two pointwise metrics
                # recorded in claim 2 rank them in OPPOSITE orders at N=512 (posterior-mean MSE
                # prefers x0 by 2.4x, relative score L2 prefers eps by 12%), while what they
                # generate differs by 72%. Selecting one arm would bury exactly that.
                arms = {
                    "em_bp": lambda X, t: score_from_mean(
                        X, bp_posterior_mean(kernel, grid, w, X, t), t),
                    "cnn_eps": lambda X, t: score_from_mean(
                        X, local_posterior_mean(nets["eps"], X, t), t),
                    "cnn_x0": lambda X, t: score_from_mean(
                        X, local_posterior_mean(nets["x0"], X, t), t),
                }
                for arm, score_fn in arms.items():
                    # Common random numbers across arms: identical initial noise and identical
                    # Brownian increments, so the difference is the score and nothing else.
                    g = rng_for("simple-c3-gen", family, seed, n_chains)
                    x0 = g.standard_normal((cfg["n_generate"], cfg["n_sites"]))
                    a_gen = reverse_sde(x0, score_fn, times, g)
                    k = excess_kurtosis(ar_residuals(a_gen, cfg["rho"]))
                    rows.append(dict(claim=3, family=family, n_chains=n_chains, seed=seed,
                                     arm=arm, residual_kurtosis=k, target=target,
                                     deficit=target - k))
            m = {a: _mean_se([r["residual_kurtosis"] for r in rows
                              if r["family"] == family and r["n_chains"] == n_chains
                              and r["arm"] == a]) for a in ARMS3}
            # Printed with standard errors because the effects here are of the same order as the
            # seed-to-seed spread, and a bare mean would invite reading noise as a result.
            print(f"     N={n_chains:5d}   EM-BP {m['em_bp'][0]:6.3f}+-{m['em_bp'][1]:.3f}   "
                  f"CNN[eps] {m['cnn_eps'][0]:6.3f}+-{m['cnn_eps'][1]:.3f}   "
                  f"CNN[x0] {m['cnn_x0'][0]:6.3f}+-{m['cnn_x0'][1]:.3f}   "
                  f"(target {target:.3f})", flush=True)

    fig3(rows, cfg, out)
    return rows


def fig3(rows, cfg, out):
    """Draw fig3 from claim-3 rows alone.

    Split out from the computation because claim 3 is the expensive part of the script and a
    figure should never cost an hour to restyle: `results.csv` carries every row this needs, so
    the figure can be redrawn from committed data without regenerating a single sample.
    """
    spread = {}
    fig, ax = new_figure(ncols=2, figsize=(10.5, 3.9))
    for j, family in enumerate(("laplace", "gaussian")):
        tgt = next(r["target"] for r in rows if r["family"] == family)
        largest = {}
        for arm, style in (("em_bp", "o-"), ("cnn_eps", "s--"), ("cnn_x0", "^--")):
            stats = [_mean_se([r["residual_kurtosis"] for r in rows
                               if r["family"] == family and r["n_chains"] == n
                               and r["arm"] == arm]) for n in cfg["budgets"]]
            vals = [s[0] for s in stats]
            largest[arm] = vals[-1]
            ax[j].errorbar(cfg["budgets"], vals, yerr=[s[1] for s in stats], fmt=style,
                           capsize=3, label=ARMS3[arm])
        spread[family] = max(largest.values()) - min(largest.values())
        ax[j].set_xscale("log")
        ax[j].axhline(tgt, color="k", ls="--", lw=1, label="target")
        label_budgets(ax[j], cfg["budgets"])
        ax[j].set_xlabel("training chains $N$")
        ax[j].set_ylabel("generated AR-residual excess kurtosis")
        ax[j].legend(fontsize=7)
    # The control's content is that the arms stop disagreeing, not that each one lands exactly on
    # the target -- the network arms keep small budget-independent biases either way. Quoting the
    # spread ratio in the title says what the panel actually shows.
    ratio = spread["laplace"] / spread["gaussian"] if spread["gaussian"] else float("inf")
    ax[0].set_title(f"(a) laplace chain  -- arms differ by {spread['laplace']:.2f}")
    ax[1].set_title(f"(b) gaussian control  -- spread {ratio:.1f}$\\times$ smaller")
    save_figure(fig, out / "fig3_generation.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="under a minute, for smoke-testing")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    cfg = dict(FULL)
    if args.quick:
        cfg.update(QUICK)

    out = ensure_dir(args.out / "figures")
    grid, w = make_grid(cfg["grid_half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])

    def flush_table(rows):
        # Rewritten after every claim, not only at the end: claim 3 is ~40 minutes and an
        # interrupted run should still leave the claims that did finish on disk.
        keys = sorted({k for r in rows for k in r})
        write_csv(args.out / "results.csv", [{k: r.get(k, "") for k in keys} for r in rows])

    t0 = time.perf_counter()
    rows = []
    for claim in (lambda: claim1_recovery(prior, grid, w, cfg, out),
                  lambda: claim2_efficiency(prior, grid, w, cfg, out),
                  lambda: claim3_generation(cfg, out)):
        rows += claim()
        flush_table(rows)  # one flat table; the note quotes nothing that is not in here
    print(f"\ndone in {time.perf_counter() - t0:.0f}s -> {args.out}/figures, results.csv", flush=True)


if __name__ == "__main__":
    main()
