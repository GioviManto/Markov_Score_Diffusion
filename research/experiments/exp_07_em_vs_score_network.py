"""Experiment 07 -- Learning the denoiser: EM through BP vs a score network.

The claim under test is Marc's: that regressing a handful of BP parameters by
expectation maximization learns the denoiser far more efficiently than training
a network to represent it. "Efficiently" is not one number, so we measure four
and let them disagree where they do.

The two estimators target the *same* object, the posterior mean
m_i(x, t) = E[a_i | x], which fixes the score through the exact OU identity.

  EM-BP     learns the transition kernel K_theta of the clean chain by exact
            EM, then computes m by belief propagation. The learned object lives
            on R x R and carries no dependence on t whatsoever.
  DSM net   learns (x, t) -> m directly by denoising score matching, the
            vanilla diffusion recipe, with no knowledge that the chain is
            Markov.

Fairness. The comparison is set up to favour the network wherever there was a
choice:
  - Data budget is the number of *clean chains* N. The network trains on paired
    (a, x) with a fresh noise draw at every gradient step, so it may consume
    unlimited noise realizations of those N chains. EM sees one noisy
    realization of each chain, at one noise level, and never sees a clean chain.
  - Both are evaluated on the same held-out chains against the same reference:
    exact grid BP under the true prior.
  - The network's objective is the one whose minimizer is exactly the target,
    so it is not handicapped by a surrogate loss.
  - Both standard parameterizations are trained and reported: noise prediction
    ("eps", the usual diffusion recipe, which carries a sqrt(Delta_t)/alpha_t
    prefactor that suppresses network error at low noise) and clean-signal
    prediction ("x0"). Resting the comparison on whichever happens to lose
    would not be evidence of anything.

Part 1 (sample efficiency). Test error vs N, the headline curve.
Part 2 (capacity). Test error vs network size at fixed N, to check the network
  is not merely undertrained or too small -- the comparison must not hinge on a
  badly chosen architecture.
Part 3 (transfer across the noise schedule). Both methods evaluated at noise
  levels *outside* the training schedule. EM-BP has no schedule to leave: its
  parameters are properties of the clean chain, and BP supplies every t exactly.
  This part was written expecting the network to break outside its schedule.
  It does not: measured across t in [0.02, 3.2] there is no cliff at the
  boundary and the EM-BP advantage does not separate in-schedule from
  out-of-schedule levels. Time enters through smooth features and the target
  moves smoothly with t, so the network handles that direction adequately --
  the difficulty is in x-space. The part is kept because the negative result
  is worth having and because it does show the x0 parameterization failing
  outright at low noise.
Part 4 (cost accounting, honestly). Training seconds, parameter counts, and
  inference seconds per chain. The last one is where BP loses, and it is
  reported as prominently as the wins.
"""

from __future__ import annotations

import time

import numpy as np

from common import apply_overrides, experiment_parser, provenance, select_parts
from src.bp_grid import make_grid
from src.denoiser import (
    bp_posterior_mean,
    dsm_posterior_mean,
    evaluate_denoiser,
    train_dsm_denoiser,
)
from src.em import fit_em
from src.kernels import MixtureInnovationKernel
from src.metrics import transition_hellinger
from src.noising import alpha_delta
from src.plotting import new_figure, save_figure
from src.protocols import one_view_groups as noisy_groups
from src.priors import LaplaceAR1
from src.utils import ensure_dir, rng_for, write_csv, write_json
from frozen_config import FROZEN

N_SITES = FROZEN.n_sites
RHO_TRUE = FROZEN.rho   # was 0.8; the two populations are now one
GRID_A = FROZEN.half_width
GRID_M = FROZEN.n_grid
T_TRAIN = tuple(FROZEN.t_grid)
N_COMPONENTS = FROZEN.n_components   # was 4; 8 is the paired-design optimum
N_TEST = FROZEN.n_heldout

# Was a hard-coded 120 in four places while the frozen config declared 400 and
# nothing read it. At 120 the innovation shape is still moving -- exp_27 puts
# the settling time at a median of 229 updates -- so the estimator was being
# reported short of convergence.
EM_ITERS = FROZEN.em_max_iters

# The iteration counts EM-BP is allowed to be stopped at, chosen on the validation
# bundle exactly as the network's parameterisation is. Log-spaced because the
# interesting structure is early: exp_27 puts rho settling near 80 and the innovation
# shape near 229, and the 120-vs-400 reversal measured on array 628943 sits between
# nseq=256 and 512, so the grid must resolve both ends.
# Base grid, extended past the standard 400-iteration cap so that a budget probe
# can be run without editing this list. `em_checkpoints` clips it to whatever cap
# it is handed, exactly as `net_checkpoints` does for the network -- the two arms
# get the same treatment, which is the whole point of this experiment.
_EM_CHECKPOINTS = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200,
                   1800, 2400, 3200)
# The tail past 1200 is not speculative. The nseq=2048 budget probe (631467)
# raised both caps 3x and freed the network -- it selects a median 20,000 of an
# allowed 60,000 and never reaches the cap -- while EM still sat at 1200 in 67%
# of cells. So EM's appetite at 2048 exceeds 1200, and at 4096 it can only be
# larger. Running nseq=4096 at a 1200 cap would reproduce the same limitation one
# size up, which is the thing the probe exists to stop.


def em_checkpoints(cap: int) -> set[int]:
    """The EM grid, clipped to a run allowed `cap` iterations.

    Mirrors `net_checkpoints`. The cap is always included, so "what the pinned
    budget would have given" stays measurable at whatever budget is in force.
    """
    return {s for s in _EM_CHECKPOINTS if s < cap} | {int(cap)}


EM_CHECKPOINTS = em_checkpoints(EM_ITERS)

# The network's training lengths, selected on the same validation bundle as
# everything else. This is the missing half of the protocol: adding EM_CHECKPOINTS
# above without this made things WORSE, not better -- EM then had its stopping
# point tuned on validation while the network was pinned to whatever NET_STEPS
# said, so the comparison tuned one arm and not the other, in EM's favour. The
# argument for selecting EM's budget is exactly the argument for selecting the
# network's, and it does not stop being true when it costs us.
#
# Log-spaced over two decades, ending at the configured budget so the pinned
# choice is always available and "what the fixed budget would have given" stays
# measurable. Wider than EM's grid because 20k steps is a long way from 500 and
# early stopping in SGD bites much earlier than the endpoint.
NET_STEPS = 20000
# Extended past 20k for the same reason EM's grid is extended past 400: at the
# largest sample size both arms often select their cap, and the only way to find
# out whether that bounds the answer is to move the grid and look.
#
# It was moved, and it does not. Tripling both caps at nseq=2048 (job 631467,
# 16 seeds, paired on seed and on validation bundle) changes the ratio by
# -0.16 +/- 0.18 and the network's error by 0.3%. Cap-selection turns out to
# report the SIGN of the remaining gain, not its size: an arm takes its last
# checkpoint whenever validation error is still falling at any rate at all.
# Keep the extended grid anyway -- it is what licenses that statement.
#
# 80k and 100k are for nseq >= 8192 and change nothing below it: the grid is
# filtered by `s < n_steps`, so a run capped at 20,000 or 60,000 sees exactly the
# checkpoints it saw before. They are here because the network's cap-hit rate
# RISES with nseq -- 33% at 2048 on a 20,000 cap, 42% at 4096 on a 60,000 one --
# so a larger sample needs a longer grid to stay off its own ceiling, and a cap
# with no checkpoints beneath it would make "selected" and "capped" the same
# event.
_NET_CHECKPOINTS = (500, 1000, 2000, 3500, 6000, 10000, 14000, 20000,
                    30000, 45000, 60000, 80000, 100000)


def net_checkpoints(n_steps: int) -> set[int]:
    """The grid, clipped to a run that trains for `n_steps`.

    Derived from the actual budget rather than fixed, because `--quick` trains
    for 1500 steps: a hard-coded grid would ask for a 20,000-step checkpoint
    that the run never reaches, and the smoke path would die on a KeyError
    instead of smoke-testing anything. `n_steps` is always included so the
    pinned budget stays available for the at-cap comparison.
    """
    return {s for s in _NET_CHECKPOINTS if s < n_steps} | {int(n_steps)}


N_VAL = 256

# Two disjoint evaluation bundles. `exp07-test` is what everything is judged on and is
# deliberately never mixed with a replicate index; `exp07-val` exists so that a choice
# between models can be made without looking at the test bundle. Neither tag appears in
# any training seed, so nothing a model is fitted on can leak into either.
TEST_TAG = "exp07-test"
VAL_TAG = "exp07-val"


def make_test_set(prior, grid, weights, t_values, n_test: int, tag: str = TEST_TAG):
    """Held-out chains plus the exact BP reference denoiser at each level."""
    rng = rng_for(tag)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_test)])
    bundle = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        bundle[t] = (X, bp_posterior_mean(prior, grid, weights, X, t))
    return A, bundle




PARAMETERIZATIONS = ("eps", "x0")

# Replicate index, mixed into every seed that draws *training* data or
# initializes a model, and deliberately NOT into the test set: replicates must
# differ in what the methods learn from and agree on what they are judged
# against, or the comparison stops being paired.
SEED_TAG = 0


def train_rng(*keys):
    return rng_for(*keys, "seed", SEED_TAG)


def score_both(kernel, nets, grid, weights, bundle, t_values, tag: dict):
    """Evaluate the BP denoiser and every trained network on the test bundle."""
    rows = []
    for t in t_values:
        X, m_ref = bundle[t]
        rows.append({**tag, "method": "em_bp", "t": t,
                     **evaluate_denoiser(
                         bp_posterior_mean(kernel, grid, weights, X, t),
                         m_ref, X, t)})
        for mode, dsm in nets.items():
            rows.append({**tag, "method": f"dsm_net_{mode}", "t": t,
                         **evaluate_denoiser(
                             dsm_posterior_mean(dsm, X, t), m_ref, X, t)})
    return rows


def train_nets(A, rng_key, hidden, n_steps, checkpoints=None):
    """One network per parameterization, on the same clean chains."""
    return {
        mode: train_dsm_denoiser(
            A, T_TRAIN, train_rng(*rng_key, mode), hidden=hidden,
            n_steps=n_steps, parameterization=mode, checkpoints=checkpoints,
        )
        for mode in PARAMETERIZATIONS
    }


# ----------------------------------------------------------------------------
# Part 1: sample efficiency
# ----------------------------------------------------------------------------

def part1_sample_efficiency(grid, weights, sizes, hidden, n_steps, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, T_TRAIN, N_TEST)
    rows, cost_rows = [], []

    for n_chains in sizes:
        rng = train_rng("exp07-p1", n_chains)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

        t0 = time.perf_counter()
        kernel, trace = fit_em(
            MixtureInnovationKernel.init(
                N_COMPONENTS, rho=0.3, var=0.8, rng=train_rng("exp07-init")
            ),
            grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=EM_ITERS,
            tol=FROZEN.em_loglik_tol,
        )
        em_seconds = time.perf_counter() - t0

        nets = train_nets(A, ("exp07-net", n_chains), hidden, n_steps)

        tag = {"n_chains": n_chains}
        rows += score_both(kernel, nets, grid, weights, bundle, T_TRAIN, tag)
        cost_rows.append({
            "n_chains": n_chains,
            "em_seconds": em_seconds,
            "em_iters": len(trace.log_evidence),
            "em_n_params": int(len(kernel.theta)),
            "em_monotone_violation": trace.monotone_violation,
            "net_n_params": nets["eps"].n_params,
            "net_grad_steps": nets["eps"].n_grad_steps,
            **{f"net_{m}_seconds": nets[m].seconds for m in PARAMETERIZATIONS},
            **{f"net_{m}_final_loss": nets[m].loss_history[-1]
               for m in PARAMETERIZATIONS},
        })

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    for method, style in (("em_bp", "o-"), ("dsm_net_eps", "s-"), ("dsm_net_x0", "^-")):
        for j, key in enumerate(("score_rel_l2", "mean_rel_l2")):
            agg = []
            for n in sizes:
                vals = [r[key] for r in rows
                        if r["method"] == method and r["n_chains"] == n]
                agg.append(float(np.mean(vals)))
            ax[j].loglog(sizes, agg, style, label=method)
    for j, ttl in enumerate(("relative score error", "relative posterior-mean error")):
        ax[j].set_xlabel("number of training chains $N$")
        ax[j].set_ylabel(ttl)
        ax[j].set_title(f"{ttl}, averaged over $t$")
        ax[j].legend()
    save_figure(fig, out / "sample_efficiency.png")
    return rows, cost_rows


# ----------------------------------------------------------------------------
# Part 5: the same comparison, with model selection moved off the test set
# ----------------------------------------------------------------------------

def part5_sample_efficiency_val(grid, weights, sizes, hidden, n_steps, out,
                                em_iters=EM_ITERS):
    """Part 1 again, with the parameterisation chosen on validation rather than on test.

    Part 1 emits `dsm_net_eps` and `dsm_net_x0` as separate rows and the better of the two
    is taken per noise level downstream. That is oracle post-selection on the evaluation
    set, and it favours the baseline rather than us: the network gets two attempts at every
    t and keeps the winner, while EM-BP gets one. The objection is real and cannot be
    answered by argument, only by measuring how much the oracle is worth.

    So: eps or x0 is chosen per noise level on a validation bundle drawn from `exp07-val`,
    disjoint from both the test bundle and every training seed, and only the winner is
    scored on test. Every row carries both numbers -- `*_selected` is the honest protocol,
    `*_oracle` is what part 1 reports -- so their ratio is the size of the selection bias.

    The fits are the *same models* part 1 evaluates: the RNG keys here are identical, so
    nothing differs between the two parts except which bundle the choice is made on. That
    is what makes the difference attributable to the protocol.

    Prediction, recorded before the run: removing an oracle only the networks enjoy should
    move the ratios in EM-BP's favour. If it moves them the other way, the validation
    bundle is too small for its selections to be anything but noise, and N_VAL is the knob.

    BOTH BUDGETS ARE NOW SELECTED, and that correction went the other way.
    --------------------------------------------------------------------
    An earlier version of this part selected EM-BP's iteration count on validation while
    the network trained to a fixed 20,000 steps. That is not a fair protocol, it is the
    same error in a new place: training length is a bias/variance knob for both arms --
    EM's held-out error has an interior minimum at every sample size (exp_29), and SGD's
    does too -- so tuning one arm's and pinning the other's favours the tuned one. It
    happened to favour us, which is precisely why it needed fixing rather than shipping.

    So the network's training length is now chosen on the same bundle, jointly with its
    parameterisation, from `net_checkpoints(n_steps)`. Each arm gets exactly one selected
    quantity: EM-BP its iteration count, the network its (parameterisation, length) pair.
    If anything the network still has the easier deal, since it selects over a
    two-dimensional grid and EM-BP over one.

    `*_at_cap` columns on both sides record what the pinned budgets would have given, so
    the effect of selection is measured on each arm separately rather than inferred from
    the ratio.
    """
    prior = LaplaceAR1(RHO_TRUE)
    _, test_bundle = make_test_set(prior, grid, weights, T_TRAIN, N_TEST, tag=TEST_TAG)
    _, val_bundle = make_test_set(prior, grid, weights, T_TRAIN, N_VAL, tag=VAL_TAG)

    rows = []
    for n_chains in sizes:
        # Same keys as part 1, so these are literally the same fitted models.
        rng = train_rng("exp07-p1", n_chains)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
        em_ckpts = em_checkpoints(em_iters)
        kernel, em_trace, ckpts = fit_em(
            MixtureInnovationKernel.init(
                N_COMPONENTS, rho=0.3, var=0.8, rng=train_rng("exp07-init")
            ),
            grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=em_iters,
            tol=FROZEN.em_loglik_tol,
            checkpoints=em_ckpts,
        )
        # Resolution and convergence certificate for the EM arm, per selected
        # checkpoint. Without these the headline ratio has no record of whether
        # its mixture components were wider than the mesh they live on -- an
        # under-resolved spike can raise the quadrature likelihood without
        # corresponding to any feature of the innovation law, so it must never be
        # able to count as evidence silently.
        em_certificate = {
            it: ckpts[it].scale_diagnostics(grid) for it in sorted(ckpts)
        }
        # Recovery at the DENSITY level, against the prior's own transition.
        #
        # Every other number here measures the fitted kernel through what it does
        # -- the score it induces. That is the quantity the diffusion model uses,
        # but it is forgiving: at moderate noise the channel has already blurred
        # away differences between visibly distinct innovation laws, so a small
        # score error is not by itself evidence that the transition was
        # recovered. This states it on the transition.
        log_k_true = prior.log_transition_matrix(grid)
        em_density = {
            it: transition_hellinger(
                ckpts[it].log_transition_matrix(grid), log_k_true, grid, weights)
            for it in sorted(ckpts)
        }
        ckpt_steps = net_checkpoints(n_steps)
        nets = train_nets(A, ("exp07-net", n_chains), hidden, n_steps,
                          checkpoints=ckpt_steps)

        for t in T_TRAIN:
            X_val, m_val = val_bundle[t]
            X_test, m_test = test_bundle[t]

            # The network's two free choices -- parameterisation and training
            # length -- are selected jointly, on validation. Selecting them
            # separately would be a different (and weaker) protocol: the best
            # stopping point is not the same for eps and x0, so a mode picked at
            # one length need not be the mode that wins at its own best length.
            candidates = [(mode, step) for mode in PARAMETERIZATIONS
                          for step in sorted(nets[mode].checkpoints)]
            val_err = {
                (mode, step): evaluate_denoiser(
                    dsm_posterior_mean(nets[mode].checkpoints[step], X_val, t, mode),
                    m_val, X_val, t)["score_rel_l2"]
                for mode, step in candidates
            }
            test_err = {
                (mode, step): evaluate_denoiser(
                    dsm_posterior_mean(nets[mode].checkpoints[step], X_test, t, mode),
                    m_test, X_test, t)
                for mode, step in candidates
            }
            chosen = min(val_err, key=val_err.get)
            oracle = min(test_err, key=lambda k: test_err[k]["score_rel_l2"])
            chosen_mode, chosen_step = chosen
            oracle_mode, oracle_step = oracle

            # The estimator's iteration count is selected the same way, on the same
            # bundle. Without this the two arms are tuned by different protocols: the
            # network gets its parameterisation chosen on validation while EM-BP is
            # pinned to whatever budget the config names. That asymmetry is not neutral
            # -- the budget is a genuine bias/variance knob here, and a fixed choice
            # favours one arm or the other depending on nseq.
            em_val = {
                it: evaluate_denoiser(
                    bp_posterior_mean(k, grid, weights, X_val, t),
                    m_val, X_val, t)["score_rel_l2"]
                for it, k in ckpts.items()
            }
            em_test = {
                it: evaluate_denoiser(
                    bp_posterior_mean(k, grid, weights, X_test, t), m_test, X_test, t)
                for it, k in ckpts.items()
            }
            em_it = min(em_val, key=em_val.get)
            em_it_oracle = min(em_test, key=lambda i: em_test[i]["score_rel_l2"])
            em = em_test[em_it]

            rows.append({
                "n_chains": n_chains,
                "t": t,
                "em_bp_score_rel_l2": em["score_rel_l2"],
                "em_bp_mean_rel_l2": em["mean_rel_l2"],
                "em_iters_selected": em_it,
                "em_iters_oracle": em_it_oracle,
                "em_iters_agrees": int(em_it == em_it_oracle),
                # What the pinned budget would have given, so the cost of selecting is
                # measurable rather than asserted.
                "em_bp_score_rel_l2_at_cap": em_test[max(em_test)]["score_rel_l2"],
                # Certificate for the checkpoint this row actually reports.
                # `em_resolved` false means the narrowest fitted component is
                # under two grid cells wide, and the row may not be cited.
                # Density-level recovery for the checkpoint this row reports.
                # There is no resolution floor -- see src/metrics.py -- so a
                # value near 0 means genuinely near-identical, not "at the
                # limit of what this metric can resolve". That used to be
                # claimed here (a stale "floor is ~4e-8" comment left over from
                # before the metric was rewritten to stop losing precision to
                # cancellation) and it was wrong even before this file existed
                # in its current form.
                "em_hellinger": em_density[em_it]["hellinger_median_interior"],
                "em_hellinger_max": em_density[em_it]["hellinger_max_interior"],
                # Parent-law-weighted mean, under the default Var(a_i)=1 parent
                # law (round-two review: the unweighted summary gives a rarely
                # visited tail parent the same vote as a typical one).
                "em_hellinger_weighted": em_density[em_it]["hellinger_weighted_mean"],
                "em_s_min_over_h": em_certificate[em_it]["s_min_over_h"],
                "em_resolved": int(em_certificate[em_it]["resolved"]),
                "em_effective_n_components":
                    em_certificate[em_it]["effective_n_components"],
                "em_min_weight": em_certificate[em_it]["min_weight"],
                "em_inner_sweeps": ckpts[em_it].inner_sweeps,
                "em_inner_converged": int(ckpts[em_it].inner_converged),
                # The gain itself, not just the flag. Auditing whether any
                # production fit ever stopped on a *decrease* previously meant
                # re-running an instrumented copy of the fit, because the
                # kernel carried `inner_q_gain` and nothing wrote it down.
                "em_inner_q_gain": float(ckpts[em_it].inner_q_gain),
                "em_outer_converged": int(em_trace.converged),
                "em_outer_stop_reason": em_trace.stop_reason,
                "net_score_rel_l2_selected": test_err[chosen]["score_rel_l2"],
                "net_mean_rel_l2_selected": test_err[chosen]["mean_rel_l2"],
                "net_score_rel_l2_oracle": test_err[oracle]["score_rel_l2"],
                "net_mean_rel_l2_oracle": test_err[oracle]["mean_rel_l2"],
                "mode_selected": chosen_mode,
                "mode_oracle": oracle_mode,
                "net_steps_selected": chosen_step,
                "net_steps_oracle": oracle_step,
                "net_steps_agrees": int(chosen_step == oracle_step),
                # The counterpart of em_bp_score_rel_l2_at_cap: what the network
                # would have scored trained to the full configured budget in the
                # mode validation picked. Without it, "early stopping helped the
                # network" is an assertion rather than a measurement.
                "net_score_rel_l2_at_cap":
                    test_err[(chosen_mode, max(ckpt_steps))]["score_rel_l2"],
                "selection_agrees": int(chosen == oracle),
                # The headline ratio under each protocol: how many times larger the
                # network's error is than EM-BP's.
                "ratio_selected": test_err[chosen]["score_rel_l2"] / em["score_rel_l2"],
                "ratio_oracle": test_err[oracle]["score_rel_l2"] / em["score_rel_l2"],
            })
            print(f"  N={n_chains:5d} t={t:4.2f} net={chosen_mode}@{chosen_step} "
                  f"em@{em_it}"
                  f"{'' if chosen == oracle else '  (net oracle differs)'}  "
                  f"ratio {rows[-1]['ratio_selected']:6.2f} vs oracle "
                  f"{rows[-1]['ratio_oracle']:6.2f}", flush=True)

    fig, ax = new_figure()
    for key, style, label in (("ratio_selected", "o-", "chosen on validation"),
                              ("ratio_oracle", "s--", "chosen on test (part 1)")):
        agg = [float(np.mean([r[key] for r in rows if r["n_chains"] == n]))
               for n in sizes]
        ax.loglog(sizes, agg, style, label=label)
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xlabel("number of training chains $N$")
    ax.set_ylabel("network error / EM-BP error")
    ax.set_title("Cost of moving model selection off the test set")
    ax.legend()
    save_figure(fig, out / "sample_efficiency_val.png")
    return rows


# ----------------------------------------------------------------------------
# Part 2: is the network merely too small or undertrained?
# ----------------------------------------------------------------------------

def part2_capacity(grid, weights, n_chains, archs, step_counts, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, T_TRAIN, N_TEST)
    rng = train_rng("exp07-p1", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

    kernel, _ = fit_em(
        MixtureInnovationKernel.init(
            N_COMPONENTS, rho=0.3, var=0.8, rng=train_rng("exp07-init")
        ),
        grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=EM_ITERS,
            tol=FROZEN.em_loglik_tol,
    )
    em_err = float(np.mean([
        evaluate_denoiser(
            bp_posterior_mean(kernel, grid, weights, bundle[t][0], t),
            bundle[t][1], bundle[t][0], t,
        )["score_rel_l2"]
        for t in T_TRAIN
    ]))

    rows = []
    for hidden in archs:
        for n_steps in step_counts:
            for mode in PARAMETERIZATIONS:
                dsm = train_dsm_denoiser(
                    A, T_TRAIN, train_rng("exp07-p2", str(hidden), n_steps, mode),
                    hidden=hidden, n_steps=n_steps, parameterization=mode,
                )
                err = float(np.mean([
                    evaluate_denoiser(
                        dsm_posterior_mean(dsm, bundle[t][0], t),
                        bundle[t][1], bundle[t][0], t,
                    )["score_rel_l2"]
                    for t in T_TRAIN
                ]))
                rows.append({
                    "n_chains": n_chains, "hidden": str(hidden),
                    "n_steps": n_steps, "parameterization": mode,
                    "net_n_params": dsm.n_params, "net_seconds": dsm.seconds,
                    "net_score_rel_l2": err,
                    "em_score_rel_l2": em_err,
                    "em_n_params": int(len(kernel.theta)),
                    "ratio_net_over_em": err / em_err,
                })

    fig, ax = new_figure()
    for hidden in archs:
        for mode, ls in zip(PARAMETERIZATIONS, ("o-", "^--")):
            sub = [r for r in rows
                   if r["hidden"] == str(hidden) and r["parameterization"] == mode]
            ax.loglog([r["n_steps"] for r in sub],
                      [r["net_score_rel_l2"] for r in sub], ls,
                      label=f"{mode}, {hidden}, {sub[0]['net_n_params']} params")
    ax.axhline(em_err, color="k", ls="--", lw=1.5,
               label=f"EM-BP ({len(kernel.theta)} params)")
    ax.set_xlabel("gradient steps")
    ax.set_ylabel("relative score error")
    ax.set_title(f"Network capacity and training budget ($N={n_chains}$)")
    ax.legend(fontsize=8)
    save_figure(fig, out / "capacity.png")
    return rows


# ----------------------------------------------------------------------------
# Part 3: transfer to unseen noise levels
# ----------------------------------------------------------------------------

def part3_transfer(grid, weights, n_chains, t_probe, hidden, n_steps, out):
    prior = LaplaceAR1(RHO_TRUE)
    _, bundle = make_test_set(prior, grid, weights, t_probe, N_TEST)
    rng = train_rng("exp07-p1", n_chains)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])

    kernel, _ = fit_em(
        MixtureInnovationKernel.init(
            N_COMPONENTS, rho=0.3, var=0.8, rng=train_rng("exp07-init")
        ),
        grid, weights, noisy_groups(A, T_TRAIN, rng), n_iters=EM_ITERS,
            tol=FROZEN.em_loglik_tol,
    )
    nets = train_nets(A, ("exp07-net", n_chains), hidden, n_steps)

    rows = score_both(kernel, nets, grid, weights, bundle, t_probe,
                      {"n_chains": n_chains})
    for r in rows:
        r["in_training_schedule"] = bool(
            any(abs(r["t"] - t) < 1e-12 for t in T_TRAIN)
        )

    fig, ax = new_figure()
    for method, style in (("em_bp", "o-"), ("dsm_net_eps", "s-"), ("dsm_net_x0", "^-")):
        sub = sorted([r for r in rows if r["method"] == method], key=lambda r: r["t"])
        ax.loglog([r["t"] for r in sub], [r["score_rel_l2"] for r in sub],
                  style, label=method)
    for t in T_TRAIN:
        ax.axvline(t, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("noise level $t$  (dotted = training schedule)")
    ax.set_ylabel("relative score error")
    ax.set_title(f"Transfer across the noise schedule ($N={n_chains}$)")
    ax.legend()
    save_figure(fig, out / "transfer.png")
    return rows


# ----------------------------------------------------------------------------
# Part 4: inference cost -- where BP loses
# ----------------------------------------------------------------------------

def part4_inference_cost(grid, weights, batch_sizes, hidden, out):
    prior = LaplaceAR1(RHO_TRUE)
    rng = train_rng("exp07-p4")
    kernel = MixtureInnovationKernel.init(
        N_COMPONENTS, rho=0.8, var=0.36, rng=train_rng("exp07-init")
    )
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(max(batch_sizes))])
    # Timing only -- neither model needs to be well fitted to measure a
    # forward pass, so the network is trained for a token number of steps.
    dsm = train_dsm_denoiser(A[:64], T_TRAIN, rng, hidden=hidden, n_steps=50)

    rows = []
    for b in batch_sizes:
        alpha, delta = alpha_delta(0.4)
        X = alpha * A[:b] + np.sqrt(delta) * rng.standard_normal((b, N_SITES))
        t0 = time.perf_counter()
        bp_posterior_mean(kernel, grid, weights, X, 0.4)
        bp_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        dsm_posterior_mean(dsm, X, 0.4)
        net_sec = time.perf_counter() - t0
        rows.append({
            "batch": b, "grid_size": len(grid),
            "bp_seconds": bp_sec, "net_seconds": net_sec,
            "bp_per_chain_ms": 1e3 * bp_sec / b,
            "net_per_chain_ms": 1e3 * net_sec / b,
            "slowdown": bp_sec / net_sec,
        })
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_07_em_vs_score_network",
        "EM-learned BP denoiser vs a denoising-score-matching network.",
    )
    args = parser.parse_args()

    quick = {
        "grid_size": 201, "sizes": (64, 256), "net_hidden": (64, 64),
        "net_steps": 1500, "archs": ((32, 32), (128, 128)),
        "step_counts": (1000, 4000), "capacity_n": 256,
        "t_probe": (0.05, 0.2, 0.8, 3.0), "inference_batches": (32, 128),
        "seed": 0, "em_iters": 40,
    }
    full = {
        # From the frozen config, not an inline literal. The list here and
        # FROZEN.sizes disagreed, and the paper quoted a third list that matched
        # neither; FROZEN.efficiency_sizes says which is right and why.
        "grid_size": GRID_M, "sizes": tuple(FROZEN.efficiency_sizes),
        # NET_STEPS, not a repeated literal: the checkpoint grid ends at it, and
        # a drift between the two would silently drop the at-cap comparison.
        "net_hidden": (128, 128), "net_steps": NET_STEPS,
        "archs": ((32, 32), (128, 128), (256, 256), (512, 512)),
        "step_counts": (1000, 5000, 20000), "capacity_n": 1024,
        "t_probe": (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6,
                    2.4, 3.2),
        "inference_batches": (32, 128, 512),
        # Both arms' caps live here so a budget probe is a --set away rather than
        # an edit -- and moving both by the same factor is the only honest probe,
        # since moving one would repeat the asymmetry this experiment was rebuilt
        # to remove. Run as job 631467 (nseq=2048, EM 400->1200, net 20k->60k):
        # the ratio moves -0.16 +/- 0.18, so the caps are selected but do not bind.
        "seed": 0, "em_iters": EM_ITERS,
    }
    cfg = apply_overrides(quick if args.quick else full, args.set)

    global SEED_TAG
    SEED_TAG = cfg["seed"]

    def p1(grid, weights, out):
        rows, cost_rows = part1_sample_efficiency(
            grid, weights, cfg["sizes"], cfg["net_hidden"], cfg["net_steps"], out)
        write_csv(out / "sample_efficiency.csv", rows)
        write_csv(out / "training_cost.csv", cost_rows)

    def p2(grid, weights, out):
        write_csv(out / "capacity.csv", part2_capacity(
            grid, weights, cfg["capacity_n"], cfg["archs"],
            cfg["step_counts"], out))

    def p3(grid, weights, out):
        write_csv(out / "transfer.csv", part3_transfer(
            grid, weights, cfg["capacity_n"], cfg["t_probe"],
            cfg["net_hidden"], cfg["net_steps"], out))

    def p4(grid, weights, out):
        write_csv(out / "inference_cost.csv", part4_inference_cost(
            grid, weights, cfg["inference_batches"], cfg["net_hidden"], out))

    def p5(grid, weights, out):
        write_csv(out / "sample_efficiency_val.csv", part5_sample_efficiency_val(
            grid, weights, cfg["sizes"], cfg["net_hidden"], cfg["net_steps"], out,
            em_iters=cfg["em_iters"]))

    parts = {
        "sample_efficiency": ("sample efficiency", p1),
        "capacity": ("network capacity and training budget", p2),
        "transfer": ("transfer across the noise schedule", p3),
        "inference_cost": ("inference cost", p4),
        "sample_efficiency_val": (
            "sample efficiency, parameterisation chosen on validation", p5),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    grid, weights = make_grid(GRID_A, cfg["grid_size"])

    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "n_sites": N_SITES, "rho_true": RHO_TRUE, "grid_half_width": GRID_A,
        "t_train": T_TRAIN, "n_components": N_COMPONENTS, "n_test": N_TEST,
        "n_val": N_VAL, "test_tag": TEST_TAG, "val_tag": VAL_TAG,
        "parameterizations": PARAMETERIZATIONS, "quick": args.quick,
        "parts": list(selected), "overrides": args.set,
        **{k: (str(v) if k == "archs" else v) for k, v in cfg.items()},
        **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(grid, weights, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
