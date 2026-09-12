"""The generalised-Gaussian kernel, and the hand-derived gradients it exists for.

Every derivative in `src/shape_kernel.py` was derived by hand, including a term that comes
from the scale depending on the shape at fixed variance -- the easiest thing in the whole
module to forget, and invisible in any check that only varies rho and q. These tests compare
all three components against central finite differences.

The family properties are checked too, because they are what makes beta a *shape* parameter
rather than another scale: variance must be exactly q for every beta, while excess kurtosis
must move from 3 (Laplace) to 0 (Gaussian).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.priors import GaussianAR1, LaplaceAR1
from src.shape_kernel import (
    GeneralizedGaussianKernel,
    generalized_gaussian_excess_kurtosis,
    generalized_gaussian_scale,
)

RHO, Q = 0.85, 1.0 - 0.85**2


@pytest.fixture(scope="module")
def grid():
    return make_grid(6.0, 121)[0]


def _numeric_variance(beta: float, q: float) -> float:
    """Variance of the generalised Gaussian by direct quadrature, not by the formula."""
    s = generalized_gaussian_scale(q, beta)
    x = np.linspace(-40.0 * s, 40.0 * s, 400001)
    logp = -np.abs(x / s) ** beta
    p = np.exp(logp - logp.max())
    p /= np.trapezoid(p, x)
    return float(np.trapezoid(x**2 * p, x))


@pytest.mark.parametrize("beta", [0.8, 1.0, 1.5, 2.0, 3.0])
def test_variance_is_q_for_every_shape(beta):
    """The defining property: beta moves shape and nothing else.

    Computed by quadrature rather than from the Gamma-ratio formula, so this checks the
    scale derivation instead of restating it.
    """
    assert _numeric_variance(beta, Q) == pytest.approx(Q, rel=2e-4)


def test_shape_endpoints_are_laplace_and_gaussian():
    assert generalized_gaussian_excess_kurtosis(1.0) == pytest.approx(3.0, abs=1e-9)
    assert generalized_gaussian_excess_kurtosis(2.0) == pytest.approx(0.0, abs=1e-9)
    # Monotone decreasing in beta across the range used by the experiment.
    ks = [generalized_gaussian_excess_kurtosis(b) for b in (0.8, 1.0, 1.5, 2.0, 3.0)]
    assert all(a > b for a, b in zip(ks, ks[1:]))


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_kernel_matches_the_known_families(grid, beta):
    """beta = 1 must reproduce the Laplace kernel and beta = 2 the Gaussian one, both of
    which exist independently in `priors`. This pins the normalisation, which a
    gradient test cannot see."""
    k = GeneralizedGaussianKernel(RHO, Q, beta).log_transition_matrix(grid)
    ref = (LaplaceAR1(RHO) if beta == 1.0 else GaussianAR1(RHO)).log_transition_matrix(grid)
    assert np.max(np.abs(k - ref)) < 1e-10


def test_columns_integrate_to_one(grid):
    """Sanity on the normaliser: each column is a density in a'.

    The residual here is quadrature, not truncation, and it is worth asserting that rather
    than a bare threshold. The density has a cusp at e = 0 for beta < 2, so the trapezoid
    rule converges more slowly than it would on a smooth integrand -- measured 2.2e-6 at
    h = 0.01 and 4.2e-7 at h = 0.005. Widening the domain at fixed spacing changes nothing
    (identical residual at A = 12 and A = 16), which is what identifies the mechanism.
    """
    def residual(half_width, n_points):
        g, w = make_grid(half_width, n_points)
        K = np.exp(GeneralizedGaussianKernel(RHO, Q, 1.4).log_transition_matrix(g))
        mass = (K * w[:, None]).sum(axis=0)
        return float(np.max(np.abs(mass[np.abs(g) <= 4.0] - 1.0)))

    coarse = residual(12.0, 2401)   # h = 0.010
    fine = residual(12.0, 4801)     # h = 0.005
    wider = residual(16.0, 6401)    # h = 0.005, larger domain

    assert coarse < 1e-5
    assert fine < coarse / 3.0                      # refining the spacing converges
    assert wider == pytest.approx(fine, rel=1e-6)   # widening the domain does nothing


@pytest.mark.parametrize("beta", [1.2, 1.6, 2.0, 2.6])
@pytest.mark.parametrize("index,name", [(0, "rho"), (1, "q"), (2, "beta")])
def test_gradient_matches_central_differences(grid, beta, index, name):
    """Every component of the hand-derived gradient, against central differences.

    The beta component is the one that matters: it carries a term from the scale's dependence
    on beta at fixed variance, and dropping that term still produces a plausible-looking
    gradient that is simply wrong. Finite differences catch it; nothing else here would.

    beta = 1 is excluded because the density has a kink in e there, so the rho-derivative is
    a subgradient rather than a derivative and central differences do not converge.

    The cusp at e = 0 is excluded for the same reason and only there. |e|^beta has unbounded
    curvature at the origin for beta < 2, so a central difference in rho straddles a point
    where no derivative exists -- the failure is in the finite-difference approximation, not
    in the analytic gradient. Measured at beta = 1.2: the relative error is 7.6e-4 when the
    single point e = 0 is included and 6.2e-9 when it is not, which is what identifies the
    cause rather than merely accommodating it. The gradient is assigned the symmetric
    subgradient 0 there, per the module docstring.
    """
    theta = np.array([RHO, Q, beta], dtype=float)
    kernel = GeneralizedGaussianKernel(*theta)
    analytic = kernel.grad_log_transition_matrix(grid)[index]

    h = 1e-6 * max(1.0, abs(theta[index]))
    up, dn = theta.copy(), theta.copy()
    up[index] += h
    dn[index] -= h
    numeric = (
        GeneralizedGaussianKernel(*up).log_transition_matrix(grid)
        - GeneralizedGaussianKernel(*dn).log_transition_matrix(grid)
    ) / (2.0 * h)

    # Compare where the density is not negligible -- far tails are dominated by cancellation
    # in the finite difference itself -- and away from the cusp at e = 0.
    logk = kernel.log_transition_matrix(grid)
    e = grid[:, None] - theta[0] * grid[None, :]
    mask = (logk > logk.max() - 25.0) & (np.abs(e) > 1e-3 * kernel.scale)

    err = np.max(np.abs(analytic[mask] - numeric[mask]))
    scale = max(1.0, float(np.max(np.abs(numeric[mask]))))
    assert err / scale < 1e-6, f"{name}: max rel error {err / scale:.2e}"


def test_rho_gradient_error_is_confined_to_the_cusp(grid):
    """The exclusion above must be a cusp, not a cover for a wrong derivative.

    If the rho-gradient were actually wrong, the discrepancy would be spread over the grid.
    This asserts it collapses by orders of magnitude the moment a single point is removed --
    the signature of a non-differentiable point, not of an algebra error.
    """
    theta = np.array([RHO, Q, 1.2])
    kernel = GeneralizedGaussianKernel(*theta)
    analytic = kernel.grad_log_transition_matrix(grid)[0]

    h = 1e-6
    up, dn = theta.copy(), theta.copy()
    up[0] += h
    dn[0] -= h
    numeric = (
        GeneralizedGaussianKernel(*up).log_transition_matrix(grid)
        - GeneralizedGaussianKernel(*dn).log_transition_matrix(grid)
    ) / (2.0 * h)

    logk = kernel.log_transition_matrix(grid)
    e = grid[:, None] - theta[0] * grid[None, :]
    near = logk > logk.max() - 25.0
    off_cusp = near & (np.abs(e) > 1e-3 * kernel.scale)

    with_cusp = np.max(np.abs(analytic - numeric)[near])
    without = np.max(np.abs(analytic - numeric)[off_cusp])
    assert with_cusp / without > 1e4


def test_beta_gradient_is_not_merely_the_explicit_terms(grid):
    """Guards the specific term most likely to be dropped.

    If the scale's dependence on beta were ignored -- i.e. if `dlogc_db` were set to zero --
    the gradient would still look reasonable. This asserts the true gradient differs
    materially from that wrong version, so the finite-difference test above is actually
    discriminating between them rather than passing either way.
    """
    from scipy.special import digamma

    k = GeneralizedGaussianKernel(RHO, Q, 1.6)
    true_grad = k.grad_log_transition_matrix(grid)[2]

    b = k.beta
    e = grid[:, None] - k.rho * grid[None, :]
    u = np.abs(e) / k.scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = np.where(u > 0, u**b, 0.0)
        log_u = np.where(u > 0, np.log(np.where(u > 0, u, 1.0)), 0.0)
    wrong = 1.0 / b + digamma(1.0 / b) / b**2 - ub * log_u

    assert np.max(np.abs(true_grad - wrong)) > 0.1


# ---------------------------------------------------------------------------
# The sampler used by exp_22, against the closed forms it must reproduce
# ---------------------------------------------------------------------------

def test_generalized_gaussian_sampler_matches_the_density():
    """`exp_22`'s sampler is the only place the family is drawn from, so it is checked
    against the analytic moments rather than against itself.

    The construction is ``sign * s * G^(1/beta)`` with ``G ~ Gamma(1/beta, 1)``, which has
    density ``beta exp(-(|x|/s)^beta) / (2 s Gamma(1/beta))``. If the exponent or the Gamma
    shape were transposed the variance would still look plausible while the kurtosis -- the
    whole point of the parameter -- would be wrong, so both are asserted.
    """
    import sys

    sys.path.insert(0, "experiments")
    from exp_22_shape_information import sample_generalized_gaussian

    rng = np.random.default_rng(20260807)
    for beta in (1.0, 1.5, 2.0, 3.0):
        k = GeneralizedGaussianKernel(0.0, Q, beta)  # rho = 0 isolates the innovation
        a = np.stack([sample_generalized_gaussian(rng, k, 3) for _ in range(30000)])
        eps = a[:, 1:].ravel()  # sites 2+ are pure innovations when rho = 0

        assert float(eps.var()) == pytest.approx(Q, rel=0.05)
        c = eps - eps.mean()
        measured_kurt = float((c**4).mean() / (c**2).mean() ** 2 - 3.0)
        assert measured_kurt == pytest.approx(
            generalized_gaussian_excess_kurtosis(beta), abs=0.25
        )
