"""Correctness tests for the strictly stationary initialisation.

Three things are being guarded, in increasing order of what they buy.

1. The power-iterated invariant density is the right object. For the Gaussian chain the
   answer is known exactly -- N(0, 1) is invariant -- so that case is a hard check with no
   tolerance to negotiate.

2. The invariant law's excess kurtosis has a closed form. For an AR(1) with innovation
   excess kurtosis k_eps and variance q, the invariant law is sum_k rho^k eps_k, fourth
   cumulants add, and

       excess kurtosis  =  k_eps q^2 / (1 - rho^4),

   since the variance is normalised to 1. That is a T1 statement about the continuous law
   checked against a T3 computation on the grid, which makes it a genuine cross-check rather
   than the iteration confirming itself. It also fails loudly if the power iteration
   converges to the wrong fixed point.

3. The two independent routes to the invariant law -- power iteration of the operator, and
   burn-in of the recursion -- agree. Neither is derived from the other.

The last test is the one that motivates the whole module: it measures the drift of the
existing N(0,1)-initialised construction directly, which is the confound in the exp_11
locality measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import grid_bp_batch, make_grid
from src.exact_scores import exact_gaussian_posterior_mean
from src.noising import alpha_delta
from src.priors import (
    GaussianAR1,
    GaussianMixtureAR1,
    LaplaceAR1,
    StudentTAR1,
    UniformAR1,
)
from src.stationary import (
    density_moments,
    invariant_log_density,
    sample_stationary_batch,
    stationary_burn_in,
)

RHO = 0.85
GRID_A = 8.0
GRID_M = 801


@pytest.fixture(scope="module")
def grid_pair():
    return make_grid(GRID_A, GRID_M)


def predicted_excess_kurtosis(prior) -> float:
    """k_eps q^2 / (1 - rho^4): the invariant law's excess kurtosis in closed form."""
    q = 1.0 - prior.rho**2
    return prior.innovation_excess_kurtosis * q**2 / (1.0 - prior.rho**4)


def test_gaussian_invariant_law_is_standard_normal(grid_pair):
    """The one case with a known answer, so it carries no tolerance argument.

    N(0,1) is exactly invariant for the Gaussian kernel, and it is also the seed the power
    iteration starts from, so a correct implementation converges in a *single* step. That
    is a sharper assertion than a loose iteration bound: it says the operator reproduces
    its own fixed point rather than merely drifting towards one.
    """
    grid, weights = grid_pair
    inv = invariant_log_density(GaussianAR1(RHO), grid, weights)

    expected = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    assert np.max(np.abs(np.exp(inv.log_density) - np.exp(expected))) < 1e-10
    assert inv.n_iter == 1


@pytest.mark.parametrize(
    "prior",
    [
        GaussianAR1(RHO),
        LaplaceAR1(RHO),
        UniformAR1(RHO),
        GaussianMixtureAR1(RHO, kappa=0.6),
        GaussianMixtureAR1(RHO, kappa=0.9),
    ],
)
def test_invariant_law_is_normalised_and_variance_matched(prior, grid_pair):
    """Unit mass, zero mean, unit variance -- the normalisation that makes the families
    comparable at all. If this failed, every cross-family comparison in the project would
    be comparing chains with different second moments.

    Mass and mean are exact to machine precision: mass because the iteration renormalises
    every step, mean because every innovation law here is symmetric and the grid is too.
    Variance is not, and its tolerance is set by the *worst* family rather than by wishful
    thinking -- 1.9e-4 for the uniform chain at this grid, whose kernel is discontinuous.
    The two tests below pin down which mechanism limits which family, so this one only has
    to catch a gross error.
    """
    grid, weights = grid_pair
    m = density_moments(invariant_log_density(prior, grid, weights).log_density,
                        grid, weights)
    assert abs(m["mass"] - 1.0) < 1e-12
    assert abs(m["mean"]) < 1e-10
    assert abs(m["var"] - 1.0) < 1e-3


@pytest.mark.parametrize(
    "prior",
    [
        GaussianAR1(RHO),
        LaplaceAR1(RHO),
        UniformAR1(RHO),
        GaussianMixtureAR1(RHO, kappa=0.6),
        GaussianMixtureAR1(RHO, kappa=0.9),
    ],
)
def test_invariant_excess_kurtosis_matches_closed_form(prior, grid_pair):
    """The cross-check that would catch a wrong fixed point.

    Deliberately excludes Student-t: its invariant law has a finite fourth moment at nu = 5,
    but the tail beyond the grid's +-8 carries a large share of it, so the discrepancy there
    measures truncation rather than the iteration. It gets its own test below, which pins
    the truncation rate instead of tolerating it.
    """
    grid, weights = grid_pair
    m = density_moments(invariant_log_density(prior, grid, weights).log_density,
                        grid, weights)
    # 2e-4 accommodates the Laplace chain's truncation at A = 8 (measured 1.34e-4). The
    # next test shows that error is the domain, not the iteration, by making it vanish.
    assert m["excess_kurtosis"] == pytest.approx(predicted_excess_kurtosis(prior), abs=2e-4)


def _kurtosis_at(prior, half_width, h=0.02):
    n_points = int(round(2.0 * half_width / h)) + 1
    grid, weights = make_grid(half_width, n_points)
    return density_moments(
        invariant_log_density(prior, grid, weights).log_density, grid, weights
    )["excess_kurtosis"]


def test_laplace_invariant_kurtosis_is_truncation_limited():
    """The exponentially tailed family is limited by the DOMAIN, not the step.

    Its invariant law has unbounded support, so a finite grid discards tail mass that the
    fourth moment weights by ``a^4``. The signature is specific enough to assert: the error
    is one-sided, since truncation can only remove mass, and it collapses when the domain
    widens at fixed step size. A two-sided or step-limited error would mean something other
    than truncation is wrong.

    Measured at h = 0.02: 7.4e-3 at A = 6, 1.3e-4 at A = 8, 1.1e-6 at A = 10, then flat at
    the quadrature floor. So the project's default A = 8 leaves a 1e-4 kurtosis error in
    this density -- four orders below the effects it is used to measure, and it is recorded
    in the experiment output rather than assumed away.
    """
    prior = LaplaceAR1(RHO)
    predicted = predicted_excess_kurtosis(prior)
    narrow = _kurtosis_at(prior, 6.0)
    default = _kurtosis_at(prior, 8.0)
    wide = _kurtosis_at(prior, 12.0)

    assert narrow < predicted            # truncation removes tail mass, never adds it
    assert default < predicted + 1e-9
    assert narrow < default < wide       # widening the domain monotonically recovers it
    assert abs(wide - predicted) < abs(narrow - predicted) / 100.0


def test_student_t_invariant_kurtosis_error_decays_as_one_over_domain():
    """The power-tailed family truncates on a law that is exactly predictable.

    For Student-t innovations with nu = 5 the invariant density has tail index 6, so the
    fourth moment discarded beyond the domain is

        int_A^inf a^4 a^-6 da  ~  1/A,

    and the kurtosis shortfall must fall as ``1/A`` rather than collapsing the way the
    Laplace chain's does. Asserting the *rate* is far stronger than asserting improvement:
    measured error times A is 2.31, 2.26, 2.25 at A = 8, 16, 32 -- constant to 3% over a
    fourfold range, which no coincidence and no partially-wrong iteration would produce.

    This is also why the heavy-tailed family is left out of the default-grid kurtosis test
    above: at A = 8 it is off by 0.29, and that is the tail, not the fixed point.
    """
    prior = StudentTAR1(RHO, nu=5.0)
    predicted = predicted_excess_kurtosis(prior)

    products = []
    for half_width in (8.0, 16.0, 32.0):
        err = predicted - _kurtosis_at(prior, half_width)
        assert err > 0.0                 # one-sided, as truncation must be
        products.append(err * half_width)

    assert max(products) / min(products) < 1.1


def test_uniform_invariant_kurtosis_is_quadrature_limited():
    """The compactly supported family is limited by the STEP, not the domain.

    A uniform innovation of half-width h_e gives an invariant law supported on
    ``[-h_e/(1-rho), h_e/(1-rho)]``, which is +-6.08 at rho = 0.85 -- inside every grid used
    here. So there is no tail to truncate and widening the domain must change nothing at
    all, which is the sharpest form this assertion can take.

    What remains is trapezoidal error on a kernel with jump discontinuities. Refining h
    reduces it, but not monotonically: the error depends on where the discontinuity falls
    between grid points, so it oscillates. That is why this compares a fourfold refinement
    rather than asserting a clean order.
    """
    prior = UniformAR1(RHO)
    assert prior.half_width / (1.0 - RHO) < 6.5

    # Flat in the domain, to machine precision.
    assert _kurtosis_at(prior, 8.0) == pytest.approx(_kurtosis_at(prior, 16.0), abs=1e-12)

    predicted = predicted_excess_kurtosis(prior)
    coarse = abs(_kurtosis_at(prior, 8.0, h=0.08) - predicted)
    fine = abs(_kurtosis_at(prior, 8.0, h=0.005) - predicted)
    assert fine < coarse / 10.0


@pytest.mark.parametrize(
    "prior", [GaussianAR1(RHO), LaplaceAR1(RHO), GaussianMixtureAR1(RHO, kappa=0.9)]
)
def test_sampler_and_power_iteration_agree(prior, grid_pair):
    """Two independent routes to the same law: burn-in of the recursion, and power
    iteration of the operator. Neither is computed from the other, so agreement is
    evidence rather than a restatement.

    The error bars have to be built rather than guessed. Sites within a chain are correlated
    at rho = 0.85, so the independent unit is the *chain*, not the sample: the standard error
    of a pooled statistic is the spread of its per-chain contribution divided by sqrt(number
    of chains). Taking `A.var(axis=1)` instead would subtract each chain's own mean and
    discard the between-chain component, which dominates here -- it understates the standard
    error of the pooled variance by a factor of about 3.5.

    Five standard errors, because this is a bug detector and not a hypothesis test: a wrong
    burn-in or a mis-scaled innovation shifts the variance by tens of percent, while honest
    sampling noise sits inside one or two.
    """
    grid, weights = grid_pair
    ref = density_moments(invariant_log_density(prior, grid, weights).log_density,
                          grid, weights)

    rng = np.random.default_rng(20260807)
    A = sample_stationary_batch(prior, rng, 3000, 16)
    n_chains = A.shape[0]

    global_mean = float(A.mean())
    chain_mean = A.mean(axis=1)
    chain_sq = ((A - global_mean) ** 2).mean(axis=1)

    se_mean = float(chain_mean.std(ddof=1) / np.sqrt(n_chains))
    se_var = float(chain_sq.std(ddof=1) / np.sqrt(n_chains))

    assert abs(global_mean - ref["mean"]) < 5.0 * se_mean
    assert abs(float(chain_sq.mean()) - ref["var"]) < 5.0 * se_var


def test_burn_in_forgets_the_initial_value():
    b = stationary_burn_in(RHO, target=1e-12)
    assert RHO**b <= 1e-12
    assert RHO ** (b - 1) > 1e-12          # not wastefully longer than needed
    assert stationary_burn_in(0.5) < b     # weaker correlation forgets sooner
    with pytest.raises(ValueError):
        stationary_burn_in(1.0)


def test_existing_construction_drifts_away_from_its_initial_law():
    """The measurement that motivates the module.

    Every prior starts at a[0] ~ N(0, 1), so site 0 is exactly Gaussian for every family --
    excess kurtosis 0 -- while a late site has moved to the invariant law, whose excess
    kurtosis is the closed form above. For the Laplace chain that is 0.483 rather than 0.

    This is why `grid_bp`'s N(0,1) default for log_mu is correct for the Gaussian chain and
    wrong for the others, and why the window estimator in exp_11 is exact for precisely the
    family the rest are compared against.
    """
    prior = LaplaceAR1(RHO)
    rng = np.random.default_rng(11)
    A = np.stack([prior.sample(rng, 60) for _ in range(40000)])

    def excess_kurtosis(col):
        c = col - col.mean()
        return float((c**4).mean() / (c**2).mean() ** 2 - 3.0)

    first = excess_kurtosis(A[:, 0])
    late = excess_kurtosis(A[:, -1])

    assert abs(first) < 0.05                                      # exactly N(0,1) by construction
    assert late == pytest.approx(predicted_excess_kurtosis(prior), abs=0.08)
    assert late - first > 0.3                                     # the drift is not marginal


def test_window_bp_matches_exact_gaussian_window_posterior(grid_pair):
    """Validates the window construction itself, on the family where it can be checked.

    A contiguous window of a Markov chain is a chain with the same kernel and the marginal of
    its left endpoint as initial law. For the Gaussian chain that marginal *is* N(0, 1), so
    window BP under the default log_mu must reproduce the exact linear-algebra posterior mean
    of the windowed Gaussian model. If it does not, the windowed estimator is wrong for every
    family and the locality experiment is measuring the bug.
    """
    grid, weights = grid_pair
    prior = GaussianAR1(RHO)
    radius, t = 4, 0.2
    width = 2 * radius + 1
    alpha, delta = alpha_delta(t)

    rng = np.random.default_rng(7)
    A = np.stack([prior.sample(rng, width) for _ in range(64)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    m_bp, _ = grid_bp_batch(grid, weights, prior.log_transition_matrix(grid), X,
                            alpha, delta)
    sigma0 = prior.covariance(width)
    m_exact = np.stack([exact_gaussian_posterior_mean(x, sigma0, alpha, delta) for x in X])

    assert np.max(np.abs(m_bp - m_exact)) < 1e-9
