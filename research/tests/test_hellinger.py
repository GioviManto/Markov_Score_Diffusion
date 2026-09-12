"""The density-level recovery metric.

Guards the properties a distance has to have before a recovery claim can rest on
it: zero exactly at identity, strictly positive off it, monotone in a parameter
that moves the density, and invariant to the normalisation convention the two
kernels happen to be written in.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel, MixtureInnovationKernel
from src.metrics import transition_hellinger
from src.utils import rng_for

RHO = 0.85


def _h(a, b, grid, weights, key="hellinger_median_interior"):
    return transition_hellinger(
        a.log_transition_matrix(grid), b.log_transition_matrix(grid), grid, weights
    )[key]


def test_identical_kernels_are_exactly_zero_apart():
    """A kernel against itself is 0, not 4e-8.

    This test used to assert the opposite -- `hellinger_max > 0.0` -- and
    justified it as pinning a floor "inherent to the metric". It was pinning an
    artefact of how the metric was written. `sqrt(1 - BC)` subtracts two numbers
    near 1 and loses the bottom 1e-15 to cancellation; the square root turns
    that into 1e-8. The equivalent sum-of-squared-sqrt-differences form never
    forms BC, so identity is exact.

    Worth keeping the story in the name: the module docstring claimed "zero
    exactly at identity" the whole time this test asserted it was positive, and
    nothing caught the contradiction because both were true of *something*.
    """
    grid, weights = make_grid(8.0, 401)
    k = GaussianAR1Kernel(RHO, 1 - RHO**2)
    out = transition_hellinger(
        k.log_transition_matrix(grid), k.log_transition_matrix(grid), grid, weights
    )
    for name, v in out.items():
        assert v == 0.0, f"{name} = {v:.3e} for a kernel against itself, want 0"


def test_near_identity_distances_are_resolved_below_the_old_floor():
    """The point of removing the floor: 1e-10 apart now reads as 1e-10 apart.

    Under `sqrt(1 - BC)` every pair closer than ~4e-8 collapsed onto the same
    number, so "identical" and "different in the tenth digit" were
    indistinguishable. Perturbing rho by 1e-9 gives a distance the old
    expression could not have represented.
    """
    grid, weights = make_grid(8.0, 401)
    a = GaussianAR1Kernel(RHO, 1 - RHO**2)
    b = GaussianAR1Kernel(RHO + 1e-9, 1 - RHO**2)
    d = _h(a, b, grid, weights, key="hellinger_max")
    assert 0.0 < d < 4e-8, f"expected a resolved sub-floor distance, got {d:.3e}"


def test_distance_grows_with_the_gap_in_rho():
    """Monotone in a parameter that moves the density, which is the whole point.

    A metric that merely separates identical from different does not support
    "the fitted transition approaches the truth"; it has to order the near
    misses too.
    """
    grid, weights = make_grid(8.0, 401)
    truth = GaussianAR1Kernel(RHO, 1 - RHO**2)
    ds = [_h(GaussianAR1Kernel(RHO - d, 1 - RHO**2), truth, grid, weights)
          for d in (0.01, 0.03, 0.10, 0.30)]
    assert all(a < b for a, b in zip(ds, ds[1:])), f"not monotone: {ds}"
    assert ds[0] > 0.0


def test_families_differing_beyond_second_moments_are_separated():
    """Variance-matched Gaussian and Laplace: identical covariance, different law.

    This is the case the score-level metrics are weakest on -- at moderate noise
    the induced scores nearly agree -- so it is exactly where a density-level
    metric has to earn its place.
    """
    grid, weights = make_grid(8.0, 401)
    q = 1 - RHO**2
    gauss = GaussianAR1Kernel(RHO, q)
    lap = LaplaceAR1Kernel(RHO, np.sqrt(q / 2.0))   # Laplace variance is 2b^2
    d = _h(lap, gauss, grid, weights)
    assert d > 0.05, f"variance-matched Gaussian and Laplace only {d:.4f} apart"


def test_invariant_to_the_normalisation_convention():
    """An unnormalised kernel must score the same as its normalised twin.

    The metric column-normalises under the grid quadrature before comparing,
    because otherwise it measures which convention each kernel was written in
    rather than how different the densities are -- a real hazard here, where the
    analytic kernels are normalised on the continuum and the MDN kernel is
    normalised on the grid.
    """
    grid, weights = make_grid(8.0, 401)
    a = GaussianAR1Kernel(RHO, 1 - RHO**2)
    b = LaplaceAR1Kernel(RHO, 0.3)
    ref = transition_hellinger(a.log_transition_matrix(grid),
                               b.log_transition_matrix(grid), grid, weights)
    # Same densities, scaled per column by an arbitrary positive factor.
    rng = rng_for("hellinger-scale")
    bump = rng.uniform(0.5, 2.0, size=len(grid))
    shifted = transition_hellinger(
        a.log_transition_matrix(grid) + np.log(bump)[None, :],
        b.log_transition_matrix(grid), grid, weights)
    for k in ref:
        assert shifted[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12)


def test_the_distance_is_bounded_in_the_unit_interval():
    """H in [0, 1] even for kernels with nearly disjoint support.

    Worth pinning because the bound is what makes the number interpretable
    without a scale: 0.3 means the same thing at every grid and every noise
    level, which is not true of the relative score error.
    """
    grid, weights = make_grid(8.0, 401)
    # Far apart: opposite autoregression and very different spread.
    a = GaussianAR1Kernel(0.95, 0.01)
    b = GaussianAR1Kernel(-0.95, 2.0)
    out = transition_hellinger(a.log_transition_matrix(grid),
                               b.log_transition_matrix(grid), grid, weights)
    for name, v in out.items():
        assert 0.0 <= v <= 1.0, f"{name} = {v} outside [0, 1]"
    assert out["hellinger_median_interior"] > 0.5, (
        f"kernels this different should be far apart, got "
        f"{out['hellinger_median_interior']:.3f}")


def test_a_fitted_mixture_approaches_the_truth_it_was_fitted_to():
    """End to end: closer parameters give a smaller density distance."""
    grid, weights = make_grid(8.0, 401)
    truth = LaplaceAR1Kernel(RHO, np.sqrt((1 - RHO**2) / 2.0))
    near = MixtureInnovationKernel.init(
        8, rho=RHO, var=1 - RHO**2, rng=rng_for("hellinger-near"))
    far = MixtureInnovationKernel.init(
        8, rho=0.2, var=3.0, rng=rng_for("hellinger-far"))
    assert _h(near, truth, grid, weights) < _h(far, truth, grid, weights)


def test_weighted_summary_matches_unweighted_under_a_uniform_parent_law():
    """A flat parent law should reduce the weighted stats to the unweighted ones.

    `weights` alone (the quadrature weights, without a parent density) are
    already non-uniform near the domain edge on some grids, so this test builds
    a `parent_law` that exactly cancels them -- 1/weights -- making the combined
    probability mass `weights * parent_law` uniform over grid points. Under that
    the weighted median and mean should coincide with the plain ones to within
    the discretisation the two summaries share.
    """
    grid, weights = make_grid(8.0, 401)
    a = GaussianAR1Kernel(RHO, 1 - RHO**2)
    b = MixtureInnovationKernel.init(8, rho=0.4, var=1.2, rng=rng_for("hellinger-uniform"))
    flat_law = 1.0 / weights
    out = transition_hellinger(
        a.log_transition_matrix(grid), b.log_transition_matrix(grid), grid, weights,
        parent_law=flat_law,
    )
    assert out["hellinger_weighted_mean"] == pytest.approx(out["hellinger_mean"], abs=1e-9)
    assert out["hellinger_weighted_median"] == pytest.approx(
        out["hellinger_median"], abs=2.0 / grid.size)


def test_weighted_summary_is_pulled_toward_where_the_parent_law_concentrates():
    """Concentrating parent mass on the worst-fit region should raise the
    weighted mean, and concentrating it on the best-fit region should lower it
    -- the property the review asked for: a single number should describe the
    regime that is actually exercised, not an equal vote per grid point
    regardless of how much mass ever lands there.

    Two Gaussian AR(1) kernels with different rho are used because their
    disagreement grows with |u| BY CONSTRUCTION and not by luck: the column at
    parent u is N(rho*u, q) against N(rho'*u, q), same variance, means rho*u
    and rho'*u, so the mean gap is |rho-rho'|*|u| and the Hellinger distance
    between two same-variance Gaussians is a strictly increasing function of
    the mean gap. So the truth about where the two kernels disagree most (the
    tails) is known analytically, not just empirically for this one draw.
    """
    grid, weights = make_grid(8.0, 401)
    truth = GaussianAR1Kernel(RHO, 1 - RHO**2)
    off = GaussianAR1Kernel(RHO - 0.3, 1 - RHO**2)

    default = transition_hellinger(
        off.log_transition_matrix(grid), truth.log_transition_matrix(grid), grid, weights)

    tail_law = np.where(np.abs(grid) > 4.0, 1.0, 1e-6)
    core_law = np.where(np.abs(grid) <= 1.0, 1.0, 1e-6)
    tail_weighted = transition_hellinger(
        off.log_transition_matrix(grid), truth.log_transition_matrix(grid), grid, weights,
        parent_law=tail_law)
    core_weighted = transition_hellinger(
        off.log_transition_matrix(grid), truth.log_transition_matrix(grid), grid, weights,
        parent_law=core_law)

    assert tail_weighted["hellinger_weighted_mean"] > default["hellinger_mean"], (
        "weighting toward a region where the fit is known to be worse should "
        "not decrease the reported distance"
    )
    assert core_weighted["hellinger_weighted_mean"] < tail_weighted["hellinger_weighted_mean"], (
        "weighting toward the well-fit core should read lower than weighting "
        "toward the poorly-fit tail -- otherwise the weighting is not doing "
        "anything the unweighted summary didn't already do"
    )


def test_default_parent_law_is_the_standard_normal():
    """The documented default -- Var(a_i) = 1 everywhere in this project -- is
    what actually gets used when no `parent_law` is passed."""
    grid, weights = make_grid(8.0, 401)
    a = GaussianAR1Kernel(RHO, 1 - RHO**2)
    b = MixtureInnovationKernel.init(6, rho=0.3, var=1.0, rng=rng_for("hellinger-default"))
    implicit = transition_hellinger(
        a.log_transition_matrix(grid), b.log_transition_matrix(grid), grid, weights)
    explicit = transition_hellinger(
        a.log_transition_matrix(grid), b.log_transition_matrix(grid), grid, weights,
        parent_law=np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi))
    assert implicit["hellinger_weighted_mean"] == explicit["hellinger_weighted_mean"]
    assert implicit["hellinger_weighted_median"] == explicit["hellinger_weighted_median"]
