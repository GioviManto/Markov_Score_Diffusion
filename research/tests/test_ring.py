"""The rotating ring: the port, the blindness theorem, and the normaliser.

Three things are pinned here.

1. **The port is faithful.** `src/ring.py` is a port of Problem 1 of
   `research/board-3problems`, which is independently audited against polar
   quadrature. The port must agree with it, not merely look like it.

2. **Marginal blindness holds.** Every single-frame marginal is psi-free. This
   is a theorem, so the test is an equality within Monte Carlo error, not a
   tolerance chosen to pass.

3. **The ring normaliser is present.** This is the one that matters most,
   because its absence is silent. Without `log_norm` the profile likelihood in
   `lam` increases without bound and the estimate pins to the top of whatever
   grid it is given -- smooth, confident, wrong. So there are two tests: the
   likelihood peaks at the truth, AND it does *not* when the normaliser is
   removed. The second exists so that deleting the term cannot make the suite
   greener.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from src.ring import (  # noqa: E402
    RingConfig,
    gauge,
    log_pt_conditional,
    log_pt_marginal,
    noise,
    sample_trajectories,
)

# The fitter lives in the experiment rather than in src/, and its stopping rule
# is a correctness question rather than a plotting detail -- see
# test_stopping_rule_does_not_depend_on_sample_size.
from exp_28_ring_em import fit_potential  # noqa: E402

T_FRAMES = 12
PSI_TRUE = np.pi / 6
LAM_TRUE = 0.05


def _dataset(n, t, psi=PSI_TRUE, cfg=None, seed=3):
    cfg = cfg or RingConfig(lam=LAM_TRUE, sigma=0.30)
    rng = np.random.default_rng(seed)
    a, _ = sample_trajectories(n, T_FRAMES, np.full(n, psi), cfg, rng)
    return noise(a, t, rng), cfg


# ---------------------------------------------------------------------------
# 1. Marginal blindness
# ---------------------------------------------------------------------------

def test_marginal_density_is_invariant_under_gauge():
    """The per-frame marginal cannot depend on psi, so gauging must not move it."""
    X, cfg = _dataset(300, 0.3)
    base = log_pt_marginal(X, 0.3, cfg)
    for wrong_psi in (0.3, 1.0, -0.7):
        rotated = log_pt_marginal(gauge(X, wrong_psi), 0.3, cfg)
        assert np.abs(rotated - base).max() < 1e-9


def test_per_frame_second_moment_does_not_depend_on_psi():
    """z_u =d z_0 + sigma sqrt(u) xi, so E|z_u|^2 is the same for every psi.

    Compared against the closed form E|z_0|^2 + 2 sigma^2 u -- two dimensions,
    hence 2 sigma^2 per step -- with a tolerance set by Monte Carlo error.
    """
    cfg = RingConfig(lam=LAM_TRUE, sigma=0.30)
    n = 40_000
    moments, errs = {}, {}
    for name, psi in [("0", 0.0), ("+30", PSI_TRUE), ("-30", -PSI_TRUE)]:
        rng = np.random.default_rng(12345)          # same draw: isolates psi
        a, _ = sample_trajectories(n, T_FRAMES, np.full(n, psi), cfg, rng)
        sq = np.linalg.norm(a, axis=2) ** 2
        moments[name] = sq.mean(axis=0)
        # Per-frame standard error. The walk spreads, so Var(|z_u|^2) grows with
        # u and a single pooled tolerance would be far too tight at late frames
        # and far too loose at early ones.
        errs[name] = sq.std(axis=0) / np.sqrt(n)

    theory = moments["0"][0] + 2.0 * cfg.sigma**2 * np.arange(T_FRAMES)

    for name, m in moments.items():
        # Two independent means, so the difference carries both errors.
        band = 4.0 * np.hypot(errs[name], errs["0"])
        assert np.all(np.abs(m - moments["0"]) < band), (
            f"psi={name} moved the marginal beyond 4 standard errors: "
            f"{np.abs(m - moments['0']) / band}"
        )
    assert np.all(np.abs(moments["0"] - theory) < 4.0 * errs["0"])


# ---------------------------------------------------------------------------
# 2. The conditional likelihood discriminates psi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.05, 0.4665])
def test_true_psi_beats_wrong_psi(t):
    X, cfg = _dataset(400, t)
    best = log_pt_conditional(X, t, PSI_TRUE, cfg).mean()
    for wrong in (0.0, -PSI_TRUE, PSI_TRUE + 0.5):
        assert best > log_pt_conditional(X, t, wrong, cfg).mean()


def test_psi_comparison_is_unaffected_by_the_constants():
    """log|A_t| and log_norm are psi-free, so psi *differences* must not move.

    This is what licenses Rungs 4b/4c to ignore them.
    """
    X, cfg = _dataset(200, 0.3)
    diffs = []
    for cfg_variant in (cfg, RingConfig(lam=LAM_TRUE, sigma=0.30)):
        a = log_pt_conditional(X, 0.3, PSI_TRUE, cfg_variant)
        b = log_pt_conditional(X, 0.3, -PSI_TRUE, cfg_variant)
        diffs.append(a - b)
    assert np.abs(diffs[0] - diffs[1]).max() < 1e-12


# ---------------------------------------------------------------------------
# 3. The normaliser -- both directions
# ---------------------------------------------------------------------------

def _profile_lambda(X, t, lams, arm="joint", subtract_norm=True):
    """Profile the likelihood in lam, optionally undoing the normaliser.

    `n_norm` is how many times the ring density is invoked: once for the joint
    model, which uses it for z_0 and propagates, and once per frame for the
    blind model, which treats the frames as independent. Getting that count
    wrong is the same bug in a different place.
    """
    out = []
    n_norm = 1 if arm == "joint" else X.shape[1]
    for lam in lams:
        cfg = RingConfig(lam=lam, sigma=0.30)
        if arm == "joint":
            ll = log_pt_conditional(X, t, PSI_TRUE, cfg).sum()
        else:
            ll = log_pt_marginal(X, t, cfg).sum()
        if not subtract_norm:
            ll += X.shape[0] * n_norm * cfg.log_norm     # reproduce the pre-fix behaviour
        out.append(ll)
    return np.array(out)


@pytest.mark.parametrize("arm", ["joint", "marginal"])
def test_lambda_profile_likelihood_peaks_at_the_truth(arm):
    """With the normaliser, the profile likelihood is peaked and finds lam.

    Both arms are checked. The blind arm is *expected* to succeed here -- the
    well is visible in every frame's radial marginal -- which is exactly the
    contrast that makes the rotation result mean something.
    """
    X, _ = _dataset(2000, 0.05)
    lams = np.array([0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.11, 0.15])
    hat = lams[int(np.argmax(_profile_lambda(X, 0.05, lams, arm)))]
    # Grid spacing near the truth is 0.005, so a step or two either way is
    # recovery at this resolution and this sample size.
    assert abs(hat - LAM_TRUE) <= 0.0101, f"{arm}: lambda_hat = {hat}"


@pytest.mark.parametrize("arm", ["joint", "marginal"])
def test_lambda_profile_is_monotone_without_the_normaliser(arm):
    """The bug this guards against, asserted so it cannot return unnoticed.

    Removing `log_norm` makes the likelihood increase without bound in lam --
    an unnormalised density simply carries more mass when the well is wider --
    so the estimate pins to the largest lam on the grid. Before the fix the
    blind arm reached lam = 3.5e6.

    Both functions are covered because the bug was fixed in
    `log_pt_conditional` first and survived in `log_pt_marginal`, where it was
    caught only by an experiment diverging.
    """
    X, _ = _dataset(2000, 0.05)
    lams = np.array([0.02, 0.03, 0.05, 0.08, 0.11, 0.15])
    ll = _profile_lambda(X, 0.05, lams, arm, subtract_norm=False)
    assert np.all(np.diff(ll) > 0), f"{arm}: expected an unnormalised monotone profile"
    assert int(np.argmax(ll)) == len(lams) - 1


# The audited sibling package, if it is checked out beside this one. Located
# relative to this file rather than by absolute path, which named a home
# directory and could only ever match one machine; BP_BOARD_CODE overrides.
BOARD_CODE = Path(os.environ.get(
    "BP_BOARD_CODE",
    Path(__file__).resolve().parents[2] / "board-3problems" / "code"))


@pytest.mark.skipif(not BOARD_CODE.exists(), reason="board-3problems not checked out")
def test_port_matches_the_audited_board_implementation():
    """The port must differ from the audited original by EXACTLY the constants.

    `board-3problems` computes the same density, audited against polar
    quadrature at 1e-9. This package adds `log|A_t| + log_norm` so that the
    result is a normalised log-density. The difference must therefore be a
    constant across the batch, and must equal that constant -- which pins both
    that nothing else changed and that the constant is the one intended.
    """
    sys.path.insert(0, str(BOARD_CODE))
    import verify_scaling as vs

    cfg = RingConfig(lam=vs.LAM, sigma=vs.SIGMA)
    t, psi, n = 0.3, PSI_TRUE, 300
    rng = np.random.default_rng(0)
    a, _ = sample_trajectories(n, T_FRAMES, np.full(n, psi), cfg, rng)
    X = noise(a, t, rng)

    diff = vs.log_pt_walk(vs.gauge_batch(X, psi), t) - log_pt_conditional(X, t, psi, cfg)
    assert diff.max() - diff.min() < 1e-12, "difference is not constant across the batch"

    A = (np.exp(-2 * t) * cfg.sigma**2
         * np.minimum.outer(np.arange(T_FRAMES), np.arange(T_FRAMES))
         + (1 - np.exp(-2 * t)) * np.eye(T_FRAMES))
    expected = float(np.linalg.slogdet(A)[1]) + cfg.log_norm
    assert abs(float(diff.mean()) - expected) < 1e-12


def test_r_star_profile_likelihood_peaks_at_the_truth():
    """The well centre is recoverable too -- the other half of the potential."""
    r_true = 1.0
    X, _ = _dataset(2000, 0.05)
    r_grid = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    ll = np.array([
        log_pt_conditional(X, 0.05, PSI_TRUE,
                           RingConfig(lam=LAM_TRUE, sigma=0.30, r_star=r)).sum()
        for r in r_grid
    ])
    hat = r_grid[int(np.argmax(ll))]
    assert abs(hat - r_true) <= 0.051, f"r_star_hat = {hat}"


def test_mixed_species_data_collapses_the_ring_under_a_single_psi():
    """A single-psi likelihood must not be handed two-species data.

    This pins the Rung 4a defect of 2026-08-17. `part_potential` generated
    trajectories at +-OMEGA and then scored all of them at one scalar psi, so
    half were evaluated under the wrong rotation. The estimator's escape is to
    shrink the ring: at r_star = 0 the ring is rotation-invariant, so a
    collapsed ring pays nothing for the wrong psi. In the sweep this showed up
    as r_star pinned to the lower clip in 52% of cells across eight seeds,
    with the blind arm apparently beating the joint arm.

    So the assertion is about the *shape of the likelihood*, not about a fitted
    value: on matched data the profile in r_star must peak at the truth, and on
    mixed data it must prefer a collapsed ring to the truth. The second half is
    what makes this a regression test rather than a restatement of
    `test_r_star_profile_likelihood_peaks_at_the_truth` -- if someone reinstates
    the two-species generation, this fails.
    """
    cfg = RingConfig(lam=LAM_TRUE, sigma=0.30)
    rng = np.random.default_rng(11)
    n, t = 800, 0.4665

    # Matched: every trajectory rotates by PSI_TRUE.
    a_one, _ = sample_trajectories(n, T_FRAMES, np.full(n, PSI_TRUE), cfg, rng)
    X_one = noise(a_one, t, rng)

    # Mixed: half at +PSI_TRUE and half at -PSI_TRUE, scored at +PSI_TRUE.
    psis = np.where(np.arange(n) < n // 2, PSI_TRUE, -PSI_TRUE)
    rng.shuffle(psis)
    a_two, _ = sample_trajectories(n, T_FRAMES, psis, cfg, rng)
    X_two = noise(a_two, t, rng)

    def profile(X, r):
        return log_pt_conditional(
            X, t, PSI_TRUE, RingConfig(lam=LAM_TRUE, sigma=0.30, r_star=r)).sum()

    collapsed, truth = 0.05, 1.0
    assert profile(X_one, truth) > profile(X_one, collapsed), (
        "matched species: the truth must beat a collapsed ring")
    assert profile(X_two, collapsed) > profile(X_two, truth), (
        "mixed species: the collapsed ring must win, which is the whole defect. "
        "If this fails the misspecification no longer bites and the test is stale.")


def test_stopping_rule_does_not_depend_on_sample_size():
    """The plateau test must ask the same question at every n.

    `_potential_loglik` sums over the rows of X, so an *absolute* plateau
    tolerance is a stricter convergence criterion the more data there is --
    eight times stricter at n=4096 than at n=512, for no statistical reason.
    Rung 4a reports how the error scales with n, so a stopping rule that is
    itself a function of n confounds the measurement with the budget. In the
    600-step sweep the marginal arm hit the cap in 95% of cells at t<=1 while
    the joint arm hit it in 57%, which made the arm comparison unreadable.

    Replicating the data k times multiplies the objective by k and leaves the
    gradient direction, the backtracking and hence the whole trajectory
    unchanged -- so the ONLY thing that can make the two fits stop at different
    steps is the stopping rule. Under a per-sequence tolerance they agree;
    under an absolute one the replicated fit sees k-times-larger improvements
    against the same threshold and runs longer.

    `plateau_tol` is passed explicitly and large, and both fits are asserted to
    have stopped BEFORE the cap. Without that guard the test is vacuous: at the
    default tolerance neither fit plateaus inside any affordable number of
    steps, both stop at the cap, and the two sizes agree no matter which rule is
    in force. That is exactly how this test passed against the defect it was
    written to catch, on 2026-08-17, before the guard was added.

    The tolerance below is calibrated, not guessed. Stopping step for this
    dataset, n against 4n, at a 400-step cap:

        plateau_tol   per-sequence rule   absolute rule
             1e-4        317   317           401   401     (both at the cap)
             1e-3         97    97           401   401     (both at the cap)
             1e-2         28    28           351   401     <- used here
             1e-1         18    18           107   215

    1e-2 is the loosest value that still leaves the fit running well past
    `plateau_window`, so the stop is a real plateau detection rather than the
    window minimum firing on its first opportunity.
    """
    cfg = RingConfig(lam=LAM_TRUE, sigma=0.30, r_star=1.0)
    rng = np.random.default_rng(5)
    n, t, cap = 120, 0.2216, 400

    a, _ = sample_trajectories(n, T_FRAMES, np.full(n, PSI_TRUE), cfg, rng)
    X = noise(a, t, rng)
    X_rep = np.concatenate([X, X, X, X])          # same law, 4x the sequences

    kw = dict(arm="joint", sigma=0.30, psi=PSI_TRUE, lam_init=0.12,
              r_init=1.30, n_steps=cap, plateau_tol=1e-2)
    _, _, h_one = fit_potential(X, t, **kw)
    _, _, h_rep = fit_potential(X_rep, t, **kw)

    assert h_one.size < cap and h_rep.size < cap, (
        f"neither fit may reach the {cap}-step cap or this test proves nothing: "
        f"got {h_one.size} and {h_rep.size}. Raise plateau_tol or the cap.")
    assert abs(h_one.size - h_rep.size) <= 1, (
        f"stopping step moved from {h_one.size} to {h_rep.size} when the same "
        "data was replicated 4x -- the plateau tolerance is still absolute, so "
        "convergence is being decided by sample size rather than by the fit.")
