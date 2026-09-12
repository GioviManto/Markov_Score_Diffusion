"""Experiment 16 -- does the score advantage survive into the generated samples?

The question, and why it is the one that matters
------------------------------------------------
Layer 5 established that EM through exact BP learns a *score* far more efficiently than a
trained network: ~10x lower relative score error against a vanilla MLP at matched data,
~4x against a locality-respecting CNN. Every one of those numbers is **pointwise** -- given
the same noisy x, how far is one denoiser from another.

Jerome's objection in the call of 29 July is that this is not what diffusion is judged on:

    "what's important with diffusion is that the error on the score, the relationship
     between the score and the distribution of data that you end up generating at the end
     of the diffusion process, is not trivial, because you accumulate the errors of the
     scores, you go backwards, and not all times are equal. So in particular, if you
     accumulate errors at intermediate times, which is what seems to be happening, you may,
     in terms of sampling, have something that's very wrong."

And he gave the reason a pointwise metric cannot settle it: at small t the denoiser barely
acts (alpha_t -> 1, "the denoiser is an identity matrix in principle"), at large t noise
dominates whatever it does, so the generated law is governed by the intermediate times. An
error averaged over the schedule sees none of that structure.

There is a concrete reason to expect the pointwise ranking *not* to transfer. exp_03 and the
cluster sweep both find the Gaussian-closure gap concentrated at **small** t -- 0.20 at
t=0.08 falling to 9e-5 at t=2.4 for the Laplace chain. That is precisely the regime Jerome
argues is cheap. So a method could look much worse pointwise and cost almost nothing in
samples.

Equally, there is a concrete reason to expect real damage: exp_05 ran the reverse SDE under
the Gaussian score on this same chain and found it reproduces second-order statistics while
flattening the innovation excess kurtosis to **0.12 against a true 2.7-2.9**. Covariance
preserved, shape destroyed.

This experiment settles which happens for the *learned* scores. It is the missing link
between "EM-BP learns a better score" and Marc's publishable claim, "we can do very efficient
learning of the denoiser using expectation maximization relative to a vanilla neural
network".

Design
------
One prior (Laplace AR(1), rho=0.85, n=32). Four score functions, all evaluated inside the
*same* reverse integrator with *common random numbers*.

Note that rho here is 0.85 while exp_07 uses 0.8, so the pointwise numbers are *not* directly
comparable across the two experiments; an earlier version of this docstring claimed they were.
The comparisons that matter here are internal -- the four score functions share a prior, a
seed and an integrator, which is what the common-random-numbers design buys.

    exact    grid BP under the true prior            -- the reference. Not a competitor:
                                                        this is the true score, so its
                                                        samples define the target.
    em_bp    grid BP under an EM-learned kernel      -- 13 parameters, fitted from N noisy
                                                        chains, never shown a clean sample.
    mlp      vanilla DSM network                     -- ~25k parameters, same N.
    cnn      weight-shared local head                -- ~6k parameters, same N.

Both networks are trained in both standard parameterizations and the better is chosen on a
fresh validation sample: eps wins at low noise, x0 at high, and reporting one alone misleads
in whichever direction is chosen. The choice is made **once per arm**, not per noise level --
switching between two independently trained networks partway through an integration gives the
integrator a score field with jump discontinuities in t that belong to neither of them.

Both are also trained on a **continuous** t schedule spanning the whole integration interval.
Training on the five levels of `t_train` and then integrating over [t_min, t_max] leaves the
network interpolating between the levels and extrapolating outside them, so the generated law
would reflect that rather than the score estimator being compared.

What is measured, and why all of it
-----------------------------------
`src/sample_metrics.py` carries the argument in full. In short, four families:

  1. MSE(m_hat, m*)  -- pure method error, the primary pointwise metric.
  2. MSE(m_hat, a)   -- the denoising risk, against its Bayes floor.
  3. relative score error -- kept for continuity, de-emphasised. It equals (1) reweighted by
     alpha_t/delta_t, which spans a factor of 65 across this schedule.
  4. distributional statistics of the generated samples -- innovation excess kurtosis
     against its bootstrap SE, histogram KL, covariance profile. The only family that sees
     accumulation.

Parts
-----
generate   The main comparison: generate from each arm at several N, measure the laws.
pointwise  The metric ladder at fixed noise levels, so the pointwise and distributional
           rankings can be put side by side and any disagreement between them attributed.
steps      Integrator control. Halving the step size must not change the ranking; if it
           does, we are measuring discretization rather than score quality.
heldout    Marginal likelihood and EM cost against mixture capacity C -- the two coordinates
           the capacity sweep was missing. Training evidence is monotone in C by nesting and
           says nothing; the held-out curve is the one that can turn over.
paired     The same capacity question on a PAIRED design: every C fitted on the same training
           set within a cell and scored on the same held-out bundle, so the reported interval
           is on the within-dataset difference rather than on two separate medians. Also
           persists the mixture scale diagnostics, since a capacity that "improves" by placing
           a component narrower than a grid cell has not improved.

The `steps` part is not optional garnish. A claim that arm A generates better samples than
arm B is worthless if the gap is an Euler-Maruyama artifact, and this is the cheapest
possible guard against reporting one.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance
from src.bp_grid import make_grid
from src.denoiser import bp_posterior_mean, dsm_posterior_mean, train_dsm_denoiser
from src.em import e_step_multi, fit_em
from src.kernels import MixtureInnovationKernel
from src.local_head import local_posterior_mean, train_local_head
from src.priors import (GaussianAR1, GaussianMixtureAR1, LaplaceAR1, StudentTAR1,
                        UniformAR1)
from src.protocols import PROTOCOLS, composite_groups, one_view_groups, sample_budget
from src.reverse import reverse_sde, time_grid
from src.sample_metrics import compare_distributions, pointwise_ladder
from src.utils import ensure_dir, rng_for, write_csv, write_json

# ---------------------------------------------------------------------------
# Settings. Overridable from the command line with --set, so a cluster run that
# scales anything stays self-describing through params.json.
# ---------------------------------------------------------------------------

SETTINGS = {
    # Data model. n=32 matches exp_07; rho does *not* (exp_07 uses 0.8), so pointwise
    # numbers are not comparable across the two experiments -- see the module docstring.
    "n_sites": 32,
    "rho": 0.85,
    # Innovation family. All are variance-matched, so they share the covariance rho^|i-j|
    # exactly and differ only beyond second moments -- which is what makes a difference in
    # the *generated* law attributable to shape rather than to correlation.
    "family": "laplace",
    # Training budgets. The interesting region is the small end, where exp_07 finds
    # EM-BP on 32 chains beating the network on 4096.
    "sizes": (32, 128, 512, 2048),
    # Noise levels the networks are trained across, and probed at.
    "t_train": (0.1, 0.2, 0.4, 0.8, 1.6),
    "t_probe": (0.1, 0.2, 0.4, 0.8, 1.6),
    # Reverse integration. t_max large enough that p_T is effectively N(0,I);
    # t_min bounded away from 0 because the score diverges as delta -> 0.
    "t_max": 3.0,
    "t_min": 0.02,
    "n_steps": 200,
    "n_generate": 2000,
    # Grid BP resolution. M=401, A=8 is the calibrated default (exp_01: worst relative
    # error 9.2e-15); it is not a free parameter and should not be lowered to save time
    # without re-running that calibration.
    "grid_m": 401,
    "grid_a": 8.0,
    # Learning budgets.
    "em_iters": 40,
    "em_components": 4,
    "net_hidden": (128, 128),
    "net_steps": 6000,
    "cnn_hidden": (64, 64),
    "cnn_steps": 6000,
    "cnn_radius": 6,
    # Replication. Every headline number is a mean over seeds with a standard error;
    # single-seed structure has dissolved twice in this project.
    "n_seed": 4,
    # Shifts the seed range, so a cluster array can run one seed per task and still cover
    # distinct seeds: task k passes seed_offset=k with n_seed=1.
    "seed_offset": 0,
    # Integrator control.
    "steps_ladder": (100, 200, 400),
    # Fresh chains for the held-out marginal likelihood. Matches the 512 the pointwise
    # part evaluates on, so the two parts describe the same evaluation population.
    "heldout_chains": 512,
    # "continuous" trains the networks log-uniformly over [t_min, t_max], matching the
    # geometric spacing of the integration grid. "discrete" reproduces the original
    # five-level protocol and is retained only so the two can be compared directly --
    # the difference between them is a reportable quantity, not a setting to forget.
    "t_schedule": "continuous",
    # "global" fixes one parameterisation per arm for the whole trajectory, chosen on a
    # fresh validation sample. "per_level" is the original nearest-probe switch, which makes
    # the integrated score field discontinuous in t between two independently trained
    # networks. Kept so the cost of the old behaviour can be measured rather than asserted.
    "param_selection": "global",
    # Observation protocol; see src/protocols.py. "composite" reproduces the original
    # behaviour (every chain noised at every level, groups treated as independent) and
    # is the default so committed outputs stay reproducible. "one_view" gives one
    # observation per latent chain, which makes the objective the exact marginal
    # likelihood rather than a composite one -- the distinction the review raised.
    "protocol": "composite",
    # Capacities compared on a paired design. Includes C=1 (a single Gaussian
    # innovation, i.e. the correctly-specified-second-order baseline) so the
    # contrast has a meaningful floor rather than starting at an already-rich model.
    "paired_components": (1, 2, 4, 8, 16, 24),
}

QUICK = {
    "sizes": (32, 128),
    "n_generate": 300,
    "n_steps": 60,
    "n_seed": 1,
    "em_iters": 8,
    "net_steps": 400,
    "cnn_steps": 400,
    "grid_m": 201,
    "steps_ladder": (60, 120),
    "heldout_chains": 64,
    "paired_components": (1, 4),
}

PARTS = ("generate", "pointwise", "steps", "heldout", "paired")


# ---------------------------------------------------------------------------
# Score functions: one builder per arm, all with the signature (X, t) -> score
# ---------------------------------------------------------------------------

def sample_chains(prior, rng: np.random.Generator, n_chains: int, n_sites: int) -> np.ndarray:
    """Draw a batch. `prior.sample` returns a single chain, so stack -- matching exp_07."""
    return np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])


def _score(X: np.ndarray, means: np.ndarray, t: float) -> np.ndarray:
    """Tweedie, applied to a batch. The single place the identity is used."""
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    return -(X - alpha * means) / delta



def reference_at(prior, rng, n_chains, n_sites, t_min):
    """A sample of ``p_{t_min}``, built the honest way: draw from ``p_0`` and noise forward.

    This replaces two earlier designs, both wrong, and the reason is worth stating because it
    is the whole methodology of the experiment.

    The reverse integration stops at ``t_min > 0`` -- it must, because the score diverges as
    ``Delta_t -> 0``. So what an arm produces is a sample of ``p_{t_min}``, *not* of ``p_0``.
    Comparing it against clean data therefore measures the stopping floor, not the score:
    independent Gaussian noise of variance ``v`` on an innovation of variance ``q`` scales
    excess kurtosis by ``(q/(q+v))^2``, which at ``rho=0.85``, ``t_min=0.02`` predicts
    **1.910**. The calibration array measured **1.91** for the exact arm. It was reproducing
    its own distribution correctly the whole time; the target was wrong.

    Applying a final denoising readout ``E[a | x, t_min]`` does not fix this, it inverts it.
    The posterior mean of a heavy-tailed prior is a shrinkage operator -- it pulls small
    values toward zero harder than large ones -- so the law of the posterior mean is *more*
    peaked than ``p_0``. Measured overshoot: 3.73, 4.34, 4.74 at ``t_min`` = 0.02, 0.05, 0.10,
    against a true 3.0, and identical at ``M=401`` and ``M=801``, which rules out grid
    resolution as the cause.

    Comparing like with like removes the problem instead of correcting for it: both sides are
    samples of ``p_{t_min}``, no estimator sits between the sampler and the metric, and the
    target moments are available in closed form (``target_innovation_moments``).
    """
    a = sample_chains(prior, rng, n_chains, n_sites)
    alpha, delta = float(np.exp(-t_min)), float(1.0 - np.exp(-2.0 * t_min))
    return alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)


def target_innovation_moments(prior, t_min: float) -> tuple[float, float]:
    """Closed-form ``(excess_kurtosis, variance)`` of the innovation under ``p_{t_min}``.

    Writing ``e_i = x_i - rho x_{i-1}`` for the noised chain,
    ``e = alpha * eps + sqrt(Delta) (z_i - rho z_{i-1})`` with the two terms independent, so
    the variance adds and the fourth cumulant is carried by the innovation alone:

        Var  = alpha^2 q + Delta (1 + rho^2)
        kurt = kurt_eps * (alpha^2 q)^2 / Var^2

    No simulation and no fitting: this is the number every arm is judged against.
    """
    alpha, delta = float(np.exp(-t_min)), float(1.0 - np.exp(-2.0 * t_min))
    v_sig = alpha ** 2 * prior.q
    v_noise = delta * (1.0 + prior.rho ** 2)
    total = v_sig + v_noise
    return float(prior.innovation_excess_kurtosis * v_sig ** 2 / total ** 2), float(total)


def noised_covariance(prior, n: int, t_min: float) -> np.ndarray:
    """``Cov`` of the chain under ``p_{t_min}``: ``alpha^2 Sigma_0 + Delta I``."""
    alpha, delta = float(np.exp(-t_min)), float(1.0 - np.exp(-2.0 * t_min))
    sigma0 = prior.rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return alpha ** 2 * sigma0 + delta * np.eye(n)


def make_exact_arm(prior, grid, weights):
    """The true score. Defines the target the other arms are judged against."""
    def fn(X, t):
        return _score(X, bp_posterior_mean(prior, grid, weights, X, float(t)), float(t))
    return fn


def make_em_arm(kernel, grid, weights):
    """Grid BP under the learned kernel -- structurally identical to `exact`.

    The *only* difference from the reference arm is which transition matrix goes in, which
    is what makes this comparison clean: any gap is attributable to the learned parameters
    and not to a different inference algorithm.
    """
    def fn(X, t):
        return _score(X, bp_posterior_mean(kernel, grid, weights, X, float(t)), float(t))
    return fn


def make_network_arm(res_eps, res_x0, better_at):
    """Oracle over the two parameterizations, chosen per noise level.

    `better_at` maps a probed t to "eps" or "x0". Giving the network the better of the two
    at every level is deliberately generous: exp_07 found eps wins at low noise and x0 at
    high, by large factors (at t=1.6, 1.19 against 0.156), so a fixed choice would make the
    baseline look bad for a reason unrelated to the question being asked.
    """
    def fn(X, t):
        t = float(t)
        res = res_eps if better_at(t) == "eps" else res_x0
        return _score(X, dsm_posterior_mean(res, X, t), t)
    return fn


def make_cnn_arm(res_eps, res_x0, better_at):
    def fn(X, t):
        t = float(t)
        res = res_eps if better_at(t) == "eps" else res_x0
        return _score(X, local_posterior_mean(res, X, t), t)
    return fn


# ---------------------------------------------------------------------------
# Fitting all four arms at one data budget
# ---------------------------------------------------------------------------

def noised_groups(A: np.ndarray, t_values, rng: np.random.Generator, protocol="composite"):
    """Observation groups for `fit_em`, under a named protocol.

    Delegates to `src.protocols`, which carries the argument in full. The short version:

    ``composite`` noises every chain at every level and hands the groups to `fit_em` as if
    they were independent latent sequences. Each marginal is correctly specified, so the
    estimator is legitimate, but the sum is a *composite* likelihood rather than the marginal
    likelihood of the generated dataset -- the effective sample size is the number of clean
    chains, not that times the number of levels. This was the only behaviour available before
    the protocol audit, so it stays the default here to keep committed outputs reproducible,
    and `composite_groups` is bit-identical to what this function used to do inline.

    ``one_view`` splits the chains disjointly across levels, one observation per latent
    chain, which makes the objective the exact marginal likelihood.
    """
    if protocol == "one_view":
        return one_view_groups(A, t_values, rng)
    if protocol in ("composite", "multi_view"):
        # multi_view shares this data layout; what differs is how the likelihood is
        # assembled, which `run_heldout` handles explicitly.
        return composite_groups(A, t_values, rng)
    raise SystemExit(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")


def training_data(prior, n_chains, seed, cfg):
    """The exact chains and noisy groups `fit_arms` fits on.

    Factored out so the held-out evaluation can reconstruct the training set by calling the
    same function, rather than replaying this generator's RNG consumption order by hand --
    which would silently diverge the first time anything upstream drew one more number.
    """
    rng = rng_for("exp16-fit", seed, n_chains)
    A = sample_chains(prior, rng, n_chains, cfg["n_sites"])
    return A, noised_groups(A, cfg["t_train"], rng, cfg["protocol"])


def fit_arms(prior, grid, weights, n_chains, seed, cfg):
    """Fit every learned arm on the *same* N noisy chains.

    The shared training set is the point: EM and the networks see identical data, so the
    comparison is about what each does with it rather than about sampling luck. Noise is
    drawn fresh per training example for the networks (the standard diffusion recipe) while
    EM sees one noisy realisation per chain -- that asymmetry favours the networks and is
    recorded rather than corrected, because correcting it would depart from how each method
    is normally used.
    """
    A, groups = training_data(prior, n_chains, seed, cfg)

    kernel, trace = fit_em(
        MixtureInnovationKernel.init(
            cfg["em_components"], rho=0.3, var=0.8,
            rng=rng_for("exp16-eminit", seed, n_chains),
        ),
        grid, weights, groups, n_iters=cfg["em_iters"],
    )

    # --- Network arms, both parameterizations.
    # Continuous-time training over the whole integration interval, not the five discrete
    # levels in t_train. The reverse integrator calls these networks at every point of
    # [t_min, t_max]; trained on five levels they interpolate between them and extrapolate
    # below 0.1 and above 1.6, so the generated-sample comparison would measure that on top
    # of the score error it exists to isolate. `t_schedule` is None only when a caller
    # deliberately wants the legacy discrete protocol for a pointwise-only run.
    t_range = None if cfg["t_schedule"] == "discrete" else (cfg["t_min"], cfg["t_max"])
    nets = {}
    for mode in ("eps", "x0"):
        nets[("mlp", mode)] = train_dsm_denoiser(
            A, cfg["t_train"], rng_for("exp16-mlp", seed, n_chains, mode),
            hidden=cfg["net_hidden"], n_steps=cfg["net_steps"], parameterization=mode,
            t_range=t_range,
        )
        nets[("cnn", mode)] = train_local_head(
            A, cfg["t_train"], cfg["cnn_radius"],
            rng_for("exp16-cnn", seed, n_chains, mode),
            hidden=cfg["cnn_hidden"], n_steps=cfg["cnn_steps"], parameterization=mode,
            t_range=t_range,
        )

    return kernel, nets, trace


def choose_parameterization(prior, grid, weights, nets, cfg, seed):
    """Pick eps or x0 per noise level, by held-out posterior-mean error.

    Selected on a *fresh* sample, not on the generated data, so the oracle cannot leak the
    comparison it is part of.
    """
    rng = rng_for("exp16-select", seed)
    A = sample_chains(prior, rng, 256, cfg["n_sites"])
    best = {}
    for t in cfg["t_probe"]:
        alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_star = bp_posterior_mean(prior, grid, weights, X, t)
        for family, predict in (("mlp", dsm_posterior_mean), ("cnn", local_posterior_mean)):
            errs = {
                mode: float(np.mean((predict(nets[(family, mode)], X, t) - m_star) ** 2))
                for mode in ("eps", "x0")
            }
            best[(family, t)] = min(errs, key=errs.get)
    return best


def _selector(best, family, t_probe, mode="global"):
    """Which parameterisation the arm uses at integrator time ``t``.

    ``global`` picks one parameterisation for the whole trajectory, by majority over the
    probed levels. ``per_level`` is the original nearest-probe lookup and is retained only
    for comparison.

    The per-level version is not a harmless convenience. During a reverse integration it
    swaps between two *independently trained* networks as ``t`` crosses the midpoint between
    probe levels, so the score field the integrator sees has jump discontinuities in ``t``
    that belong to neither network. A generated sample then reflects an estimator nobody
    fitted. Choosing once, on validation data, keeps the arm a single well-defined score
    model -- which is the only thing a reverse-SDE comparison can meaningfully be about.
    """
    if mode not in ("global", "per_level"):
        raise ValueError(f"Unknown parameterisation-selection mode {mode!r}.")

    if mode == "global":
        votes = [best[(family, u)] for u in t_probe]
        # Deterministic tie-break: count descending, then lexicographic. `max(set(votes),
        # key=votes.count)` would resolve a tie by set iteration order, which CPython salts
        # per process for str -- the identical nondeterminism that was just removed from
        # exp_18's seeding, and that hpc/bocconi_p0.sbatch has a dedicated job to catch.
        # A tie is impossible with the default five probe levels and two parameterisations,
        # but --set t_probe=... with an even count is supported and would hit it.
        winner = sorted(set(votes), key=lambda v: (-votes.count(v), v))[0]
        return lambda t: winner

    def fn(t):
        nearest = min(t_probe, key=lambda u: abs(u - t))
        return best[(family, nearest)]
    return fn


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def run_generate(prior, grid, weights, cfg, out_dir):
    """Generate from every arm and measure the resulting laws."""
    rows = []
    times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps"])
    n = cfg["n_sites"]
    sigma_true = noised_covariance(prior, n, cfg["t_min"])
    kurt_target, var_target = target_innovation_moments(prior, cfg["t_min"])
    print(f"  target for p_t at t_min={cfg['t_min']}: excess kurtosis {kurt_target:.4f}, "
          f"innovation variance {var_target:.4f}", flush=True)

    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            kernel, nets, trace = fit_arms(prior, grid, weights, n_chains, seed, cfg)
            best = choose_parameterization(prior, grid, weights, nets, cfg, seed)

            arms = {
                "exact": make_exact_arm(prior, grid, weights),
                "em_bp": make_em_arm(kernel, grid, weights),
                "mlp": make_network_arm(nets[("mlp", "eps")], nets[("mlp", "x0")],
                                        _selector(best, "mlp", cfg["t_probe"], cfg["param_selection"])),
                "cnn": make_cnn_arm(nets[("cnn", "eps")], nets[("cnn", "x0")],
                                    _selector(best, "cnn", cfg["t_probe"], cfg["param_selection"])),
            }

            # A reference sample from the forward model -- the yardstick. Never a
            # reverse-generated one, which would fold the integrator's error into the target.
            a_ref = reference_at(prior, rng_for("exp16-ref", seed),
                                 cfg["n_generate"], n, cfg["t_min"])

            for name, fn in arms.items():
                # Common random numbers: identical initial noise and identical Brownian
                # increments across arms, so differences are the score and nothing else.
                rng = rng_for("exp16-gen", seed, n_chains)
                x_init = rng.standard_normal((cfg["n_generate"], n))
                a_gen = reverse_sde(x_init, fn, times, rng)

                c = compare_distributions(
                    a_gen, a_ref, prior.rho, sigma_true,
                    innov_kurtosis_true=kurt_target,
                    innov_variance_true=var_target, name=name, seed=seed,
                )
                rows.append({
                    "n_chains": n_chains, "seed": seed, "arm": name,
                    "innov_kurtosis": c.innov_kurtosis,
                    "innov_kurtosis_se": c.innov_kurtosis_se,
                    "kurtosis_gap_in_se": c.kurtosis_gap_in_se(),
                    "innov_variance": c.innov_variance,
                    "innov_kl": c.innov_kl,
                    "cov_frobenius_rel": c.cov_frobenius_rel,
                    "cov_worst_lag_abs": c.cov_worst_lag_abs,
                    "marginal_var": c.marginal_var,
                    "notes": "; ".join(c.notes),
                })
                print(f"  N={n_chains:5d} seed={seed} {name:6s} "
                      f"kurt={c.innov_kurtosis:6.3f}+-{c.innov_kurtosis_se:.3f} "
                      f"(true {c.innov_kurtosis_true:.2f}, gap {c.kurtosis_gap_in_se():6.1f} se)  "
                      f"KL={c.innov_kl:.4f}  cov_lag={c.cov_worst_lag_abs:.4f}", flush=True)

    write_csv(out_dir / "generation.csv", rows)
    return rows


def run_pointwise(prior, grid, weights, cfg, out_dir):
    """The metric ladder at fixed noise levels, for side-by-side with `generate`."""
    rows = []
    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            kernel, nets, _ = fit_arms(prior, grid, weights, n_chains, seed, cfg)
            best = choose_parameterization(prior, grid, weights, nets, cfg, seed)

            rng = rng_for("exp16-point", seed, n_chains)
            A = sample_chains(prior, rng, 512, cfg["n_sites"])
            for t in cfg["t_probe"]:
                alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
                X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
                m_star = bp_posterior_mean(prior, grid, weights, X, t)

                preds = {
                    "em_bp": bp_posterior_mean(kernel, grid, weights, X, t),
                    "mlp": dsm_posterior_mean(nets[("mlp", best[("mlp", t)])], X, t),
                    "cnn": local_posterior_mean(nets[("cnn", best[("cnn", t)])], X, t),
                }
                for name, m_hat in preds.items():
                    d = pointwise_ladder(m_hat, m_star, A, alpha, delta)
                    rows.append({"n_chains": n_chains, "seed": seed, "arm": name,
                                 "t": t, **d})
    write_csv(out_dir / "pointwise.csv", rows)
    return rows


def run_heldout(prior, grid, weights, cfg, out_dir):
    """Marginal likelihood and EM cost against innovation-mixture capacity C.

    The capacity sweep measured pointwise error and generated statistics but never the two
    quantities that decide whether more components is *justified* rather than merely better
    on those two axes. Both were already being computed and thrown away: `fit_em` records
    the exact marginal log-likelihood and the wall clock of every iteration in its EMTrace,
    and `fit_arms` returns it.

    Two separations matter before reading the numbers.

    **Training evidence cannot fall as C grows** -- a C-component mixture contains every
    (C-1)-component one -- so that curve carries no model-selection information at all. Only
    the held-out curve can turn over, and whether it does is the question this part exists to
    answer. The held-out chains are drawn fresh and noised by `noised_groups`, the same
    construction the training set uses, so the two log-likelihoods are comparable per edge.

    **Cost does not obviously scale with C.** The E-step is O(N n M^2) and does not involve C
    anywhere; only the M-step does. So seconds per iteration should be near-flat in C, and a
    rise would mean the M-step dominates -- which would change the cost statement in the
    note rather than decorate it.

    The training evidence here is recomputed at the *returned* kernel. `fit_em` appends
    log_evidence before each M-step, so `trace.log_evidence[-1]` belongs to the parameters
    one M-step behind the kernel that is actually reported; comparing that against a
    held-out number evaluated at the final kernel would compare two different models.
    """
    rows, trace_rows = [], []
    n_comp = cfg["em_components"]
    # rho (1) + weights (C, one simplex constraint) + locations (C) + scales (C).
    n_params = 3 * n_comp

    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            kernel, _, trace = fit_arms(prior, grid, weights, n_chains, seed, cfg)

            for it, (le, sec) in enumerate(zip(trace.log_evidence, trace.seconds)):
                trace_rows.append({
                    "n_components": n_comp, "n_chains": n_chains, "seed": seed,
                    "iter": it, "log_evidence": le, "seconds": sec,
                })

            log_k = kernel.log_transition_matrix(grid)
            _, train_groups = training_data(prior, n_chains, seed, cfg)
            train = e_step_multi(grid, weights, log_k, train_groups)

            rng = rng_for("exp16-heldout", seed, n_chains)
            A_new = sample_chains(prior, rng, cfg["heldout_chains"], cfg["n_sites"])
            held = e_step_multi(
                grid, weights, log_k, noised_groups(A_new, cfg["t_train"], rng)
            )

            rows.append({
                "n_components": n_comp,
                "n_chains": n_chains,
                "seed": seed,
                "n_params": n_params,
                "n_iters": len(trace.log_evidence),
                "monotone_violation": trace.monotone_violation,
                "train_log_evidence": train.log_evidence,
                "train_log_evidence_per_edge": train.log_evidence / train.n_edges,
                "train_edges": train.n_edges,
                "heldout_log_evidence": held.log_evidence,
                "heldout_log_evidence_per_edge": held.log_evidence / held.n_edges,
                "heldout_edges": held.n_edges,
                "bic": -2.0 * train.log_evidence + n_params * float(np.log(train.n_edges)),
                "em_seconds_total": float(np.sum(trace.seconds)),
                "em_seconds_per_iter": float(np.mean(trace.seconds)),
                "fitted_rho": float(kernel.rho),
                **kernel.innovation_moments,
                **sample_budget(cfg["protocol"], n_chains, cfg["n_sites"],
                                len(cfg["t_train"])),
                # Whether the fitted mixture is resolved by the grid. The C=12->16
                # improvement cannot be read as real while the narrowest component is
                # narrower than a cell.
                **kernel.scale_diagnostics(grid),
            })
            print(f"  C={n_comp:2d} N={n_chains:5d} seed={seed} "
                  f"train/edge={rows[-1]['train_log_evidence_per_edge']:+.5f} "
                  f"held/edge={rows[-1]['heldout_log_evidence_per_edge']:+.5f}  "
                  f"{rows[-1]['em_seconds_per_iter']:.2f}s/iter  "
                  f"viol={trace.monotone_violation:.2e}", flush=True)

    write_csv(out_dir / "em_trace.csv", trace_rows)
    write_csv(out_dir / "heldout_evidence.csv", rows)
    return rows


def run_paired(prior, grid, weights, cfg, out_dir):
    """Capacity and data budget on a **paired** design, with intervals on differences.

    The capacity sweep so far reports separate medians at each C over three seeds, and the
    C = 12 -> 16 improvement it shows is smaller than the spread between seeds. Separate
    medians cannot settle that: most of the variance is the training draw, and it is shared
    between capacities, so comparing marginal summaries throws away exactly the pairing that
    would make the difference visible.

    Here every capacity is fitted on the **same** training set within a (budget, seed) cell
    and evaluated on the **same** held-out bundle, so the per-cell difference between two
    capacities is a within-dataset contrast. The standard error of the mean difference is
    then what says whether C = 16 beats C = 12, and it is typically far smaller than the
    standard error of either arm alone.

    Also persists, per fit, the quantities the audit asked for and that decide whether a
    small gain is real: the fitted mixture's minimum component scale relative to the grid
    spacing, its effective component count, and its column-mass residual. A capacity that
    "improves" by placing a component narrower than a grid cell has not improved.
    """
    rows = []
    t_eval = cfg["t_probe"]

    for n_chains in cfg["sizes"]:
        for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
            # One training set and one evaluation set per cell, shared by every capacity.
            A, groups = training_data(prior, n_chains, seed, cfg)
            rng_e = rng_for("exp16-paired-eval", seed, n_chains)
            A_eval = sample_chains(prior, rng_e, cfg["heldout_chains"], cfg["n_sites"])

            bundles = {}
            for t in t_eval:
                alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
                X = alpha * A_eval + np.sqrt(delta) * rng_e.standard_normal(A_eval.shape)
                bundles[t] = (X, alpha, delta,
                              bp_posterior_mean(prior, grid, weights, X, t))

            for n_comp in cfg["paired_components"]:
                kernel, trace = fit_em(
                    MixtureInnovationKernel.init(
                        n_comp, rho=0.3, var=0.8,
                        rng=rng_for("exp16-paired-init", seed, n_chains, n_comp),
                    ),
                    grid, weights, groups, n_iters=cfg["em_iters"],
                )
                log_k = kernel.log_transition_matrix(grid)
                held = e_step_multi(
                    grid, weights, log_k,
                    noised_groups(A_eval, cfg["t_train"],
                                  rng_for("exp16-paired-held", seed, n_chains),
                                  cfg["protocol"]),
                )

                for t in t_eval:
                    X, alpha, delta, m_star = bundles[t]
                    m_hat = bp_posterior_mean(kernel, grid, weights, X, t)
                    rows.append({
                        "n_chains": n_chains,
                        "seed": seed,
                        "n_components": n_comp,
                        "t": t,
                        "mse_vs_bayes": float(np.mean((m_hat - m_star) ** 2)),
                        "heldout_log_evidence_per_edge":
                            held.log_evidence / held.n_edges,
                        "em_seconds_total": float(np.sum(trace.seconds)),
                        "fitted_rho": float(kernel.rho),
                        **kernel.innovation_moments,
                        **kernel.scale_diagnostics(grid),
                    })
                print(f"  N={n_chains:5d} seed={seed} C={n_comp:2d} "
                      f"held/edge={rows[-1]['heldout_log_evidence_per_edge']:+.6f} "
                      f"s_min/h={rows[-1]['s_min_over_h']:.2f} "
                      f"eff_C={rows[-1]['effective_n_components']:.2f}", flush=True)

    write_csv(out_dir / "paired_capacity.csv", rows)

    # Paired contrasts against the smallest capacity, with a standard error over cells.
    base = min(cfg["paired_components"])
    contrasts = []
    for n_chains in cfg["sizes"]:
        for n_comp in cfg["paired_components"]:
            if n_comp == base:
                continue
            diffs = []
            for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
                def pick(c):
                    v = [r["mse_vs_bayes"] for r in rows
                         if r["n_chains"] == n_chains and r["seed"] == seed
                         and r["n_components"] == c]
                    return float(np.mean(v)) if v else np.nan
                diffs.append(pick(n_comp) - pick(base))
            d = np.asarray(diffs, dtype=float)
            n = len(d)
            contrasts.append({
                "n_chains": n_chains,
                "n_components": n_comp,
                "baseline_components": base,
                "n_pairs": n,
                "mean_paired_diff": float(np.mean(d)),
                "se_paired_diff": float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                # Negative and beyond two standard errors means the extra capacity helped.
                "significant": bool(
                    n > 1 and abs(np.mean(d)) > 2.0 * np.std(d, ddof=1) / np.sqrt(n)
                ),
            })
    write_csv(out_dir / "paired_contrasts.csv", contrasts)
    return rows


def run_steps(prior, grid, weights, cfg, out_dir):
    """Integrator control: does the ranking survive a change of step size?

    If halving the step changes which arm wins, the comparison is measuring
    Euler-Maruyama discretization rather than score quality, and nothing in `generate`
    should be believed.
    """
    rows = []
    n = cfg["n_sites"]
    sigma_true = noised_covariance(prior, n, cfg["t_min"])
    kurt_target, var_target = target_innovation_moments(prior, cfg["t_min"])
    n_chains = cfg["sizes"][-1]

    kernel, nets, _ = fit_arms(prior, grid, weights, n_chains, 0, cfg)
    best = choose_parameterization(prior, grid, weights, nets, cfg, 0)
    arms = {
        "exact": make_exact_arm(prior, grid, weights),
        "em_bp": make_em_arm(kernel, grid, weights),
        "mlp": make_network_arm(nets[("mlp", "eps")], nets[("mlp", "x0")],
                                _selector(best, "mlp", cfg["t_probe"], cfg["param_selection"])),
        "cnn": make_cnn_arm(nets[("cnn", "eps")], nets[("cnn", "x0")],
                            _selector(best, "cnn", cfg["t_probe"], cfg["param_selection"])),
    }
    a_ref = reference_at(prior, rng_for("exp16-ref", 0), cfg["n_generate"], n, cfg["t_min"])

    for n_steps in cfg["steps_ladder"]:
        times = time_grid(cfg["t_max"], cfg["t_min"], n_steps)
        for name, fn in arms.items():
            rng = rng_for("exp16-steps", n_steps)
            x_init = rng.standard_normal((cfg["n_generate"], n))
            a_gen = reverse_sde(x_init, fn, times, rng)
            c = compare_distributions(
                a_gen, a_ref, prior.rho, sigma_true,
                innov_kurtosis_true=kurt_target,
                innov_variance_true=var_target, name=name, seed=0,
            )
            rows.append({"n_steps": n_steps, "arm": name,
                         "innov_kurtosis": c.innov_kurtosis,
                         "innov_kurtosis_se": c.innov_kurtosis_se,
                         "innov_kl": c.innov_kl,
                         "cov_worst_lag_abs": c.cov_worst_lag_abs})
            print(f"  steps={n_steps:4d} {name:6s} kurt={c.innov_kurtosis:6.3f} "
                  f"KL={c.innov_kl:.4f}", flush=True)

    write_csv(out_dir / "steps.csv", rows)
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    parser = experiment_parser("exp_16_sampling_validation", __doc__)
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = PARTS if args.only is None else tuple(p.strip() for p in args.only.split(","))
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown part(s): {sorted(unknown)}; choose from {PARTS}")

    out_dir = ensure_dir(args.output_dir)
    grid, weights = make_grid(cfg["grid_a"], cfg["grid_m"])
    priors = {"laplace": LaplaceAR1, "student": StudentTAR1, "uniform": UniformAR1,
              "mixture": GaussianMixtureAR1, "gaussian": GaussianAR1}
    if cfg["family"] not in priors:
        raise SystemExit(f"unknown family {cfg['family']!r}; choose from {sorted(priors)}")
    prior = priors[cfg["family"]](cfg["rho"])

    for part in parts:
        print(f"\n=== part: {part} ===", flush=True)
        if part == "generate":
            run_generate(prior, grid, weights, cfg, out_dir)
        elif part == "pointwise":
            run_pointwise(prior, grid, weights, cfg, out_dir)
        elif part == "steps":
            run_steps(prior, grid, weights, cfg, out_dir)
        elif part == "heldout":
            run_heldout(prior, grid, weights, cfg, out_dir)
        elif part == "paired":
            run_paired(prior, grid, weights, cfg, out_dir)

    write_json(out_dir / f"params_{'_'.join(parts)}.json",
               {"settings": cfg, "parts": list(parts), **provenance()})


if __name__ == "__main__":
    main()
