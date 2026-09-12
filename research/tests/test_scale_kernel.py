"""The scale-mixture kernel: does it learn what the linear-AR family cannot?

The decisive test is `test_recovers_magnitude_dependence`, which fits both
families to data whose child *spread* depends on the parent magnitude and checks
that this kernel recovers it while `MixtureInnovationKernel` returns a flat
conditional standard deviation. That is not a criticism of the mixture kernel --
a location-shift family cannot represent a scale effect, by construction -- it is
the measurement that justifies adding a second family at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import clean_statistics
from src.kernels import MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel, linear_ar_magnitude_ratio
from src.utils import rng_for

_LOG_2PI = float(np.log(2.0 * np.pi))


def _q_value(kernel, xi, grid):
    """Q(theta) = sum_{kj} Xi[k,j] log K(u_k | u_j), the ECM objective."""
    return float((xi * kernel.log_transition_matrix(grid)).sum())


def _pairs_to_xi(grid, parent, child):
    return clean_statistics(grid, np.stack([parent, child], axis=1)).xi


def _empirical_ratio(parent, child):
    q = np.quantile(np.abs(parent), [0.25, 0.75])
    lo = child[np.abs(parent) <= q[0]].std()
    hi = child[np.abs(parent) >= q[1]].std()
    return float(hi / lo)


# ----------------------------------------------------------------------------
# The density
# ----------------------------------------------------------------------------

def test_column_mass_deficit_is_truncation_and_responds_to_the_domain():
    """Columns integrate to 1 up to *truncation*, not up to quadrature error.

    The residual is measured the way the project measures it elsewhere
    (advisor document question A, Fig. 2): truncation responds to the domain A
    and not to the grid size N_g, while quadrature responds to the spacing. A
    scale mixture makes this visible because its widest component is wide -- at
    var = 0.6 the broadest component has standard deviation 1.7, so on a +-9
    domain a parent at u = 4 still has ~2e-5 of its child mass off-grid.
    """
    rng = rng_for("scale-kernel-norm")
    k = ScaleMixtureKernel.init(4, rho=0.5, var=0.6, rng=rng)

    def deficit(half_width, size):
        grid, weights = make_grid(half_width, size)
        mass = (np.exp(k.log_transition_matrix(grid)) * weights[:, None]).sum(axis=0)
        return float(np.max(np.abs(mass[np.abs(grid) <= 4.0] - 1.0)))

    coarse, fine = deficit(9.0, 801), deficit(9.0, 1601)
    # Doubling N_g at fixed domain moves the residual by ~2%: it is not
    # quadrature error. Enlarging the domain kills it by nine orders of
    # magnitude: it is truncation.
    assert abs(coarse - fine) < 0.03 * coarse
    assert coarse < 1e-4
    assert deficit(12.0, 1201) < 1e-8
    assert deficit(14.0, 1201) < 1e-11


def test_single_component_is_exactly_gaussian_ar1():
    grid, _ = make_grid(8.0, 401)
    k = ScaleMixtureKernel(
        rho=np.array([0.7]), s2=np.array([0.4]),
        beta=np.zeros(1), gamma=np.zeros(1),
    )
    resid = grid[:, None] - 0.7 * grid[None, :]
    want = -0.5 * (_LOG_2PI + np.log(0.4)) - 0.5 * resid**2 / 0.4
    assert np.max(np.abs(k.log_transition_matrix(grid) - want)) < 1e-12


def test_gate_equals_a_gaussian_scale_mixture_responsibility():
    """The quadratic logit is not an arbitrary parametrisation: it is exactly the
    responsibility of a zero-mean Gaussian scale mixture, which is what the
    docstring claims and what makes the fitted gate interpretable."""
    grid, _ = make_grid(6.0, 301)
    pi = np.array([0.6, 0.4])
    tau2 = np.array([0.5, 3.0])
    beta = np.log(pi) - 0.5 * np.log(tau2)
    gamma = -0.5 / tau2
    k = ScaleMixtureKernel(
        rho=np.zeros(2), s2=np.ones(2),
        beta=beta - beta[0], gamma=gamma - gamma[0],
    )
    log_r = (np.log(pi)[None, :] - 0.5 * np.log(2 * np.pi * tau2)[None, :]
             - 0.5 * (grid**2)[:, None] / tau2[None, :])
    want = np.exp(log_r - log_r.max(1, keepdims=True))
    want = want / want.sum(1, keepdims=True)
    assert np.max(np.abs(k.gate(grid) - want)) < 1e-12


# ----------------------------------------------------------------------------
# The M-step
# ----------------------------------------------------------------------------

def test_m_step_increases_q_every_iteration():
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-monotone")
    p = rng.standard_normal(40000)
    c = 0.5 * p + (0.4 + 0.8 * np.abs(p)) * rng.standard_normal(40000)
    xi = _pairs_to_xi(grid, p, c)

    k = ScaleMixtureKernel.init(3, rho=0.2, var=1.0, rng=rng)
    q_prev = _q_value(k, xi, grid)
    for _ in range(8):
        k = k.m_step(xi_stats(xi), grid)
        q_now = _q_value(k, xi, grid)
        assert q_now >= q_prev - 1e-6 * abs(q_prev)
        q_prev = q_now


def xi_stats(xi):
    from src.em import ExpectedStatistics

    return ExpectedStatistics(
        xi=xi, site1=np.zeros(xi.shape[0]), log_evidence=float("nan"),
        n_edges=int(xi.sum()), n_chains=1,
    )


def test_recovers_magnitude_dependence_where_the_mixture_cannot():
    """The reason this module exists.

    Data with a parent-magnitude-dependent child spread. The scale-mixture kernel
    must recover a conditional standard deviation that grows with |parent|; the
    linear-AR mixture must return a flat one whatever its component count.
    """
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-recover")
    n = 200000
    p = rng.standard_normal(n)
    c = 0.35 * p + (0.35 + 0.85 * np.abs(p)) * rng.standard_normal(n)
    target = _empirical_ratio(p, c)
    assert target > 2.0, "test data must actually have magnitude dependence"

    xi = _pairs_to_xi(grid, p, c)

    scale_k = ScaleMixtureKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        scale_k = scale_k.m_step(xi_stats(xi), grid)

    mix_k = MixtureInnovationKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        mix_k = mix_k.m_step(xi_stats(xi), grid)

    got = scale_k.magnitude_ratio(grid)
    null = linear_ar_magnitude_ratio(0.35)
    assert got > null + 1.0, (
        f"scale mixture recovered {got:.3f}, barely above the linear-AR null "
        f"{null:.3f}; target (empirical) is {target:.3f}"
    )

    # The mixture kernel's conditional variance is independent of the parent by
    # construction; confirm numerically rather than by assertion.
    logk = mix_k.log_transition_matrix(grid)
    k_mat = np.exp(logk)
    k_mat = k_mat / np.maximum(k_mat.sum(axis=0, keepdims=True), 1e-300)
    mean = (k_mat * grid[:, None]).sum(axis=0)
    var = (k_mat * (grid[:, None] - mean[None, :]) ** 2).sum(axis=0)
    near = np.abs(grid) <= 2.0
    flat = float(np.sqrt(var[near]).max() / np.sqrt(var[near]).min())
    assert flat < 1.15, f"mixture kernel unexpectedly varied its scale: {flat}"

    # And the scale kernel must beat it on the objective it is fitted to.
    assert _q_value(scale_k, xi, grid) > _q_value(mix_k, xi, grid)


def test_linear_ar_null_matches_simulation():
    """The null has to be right, because the headline claim is measured against it."""
    rng = rng_for("scale-kernel-null")
    n = 2000000
    for rho in (0.0, 0.148, 0.452, 0.6):
        p = rng.standard_normal(n)
        c = rho * p + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        assert abs(_empirical_ratio(p, c) - linear_ar_magnitude_ratio(rho)) < 0.01


def test_gaussian_control_reproduces_the_linear_null_and_no_more():
    """The negative control the project uses everywhere.

    On genuinely Gaussian AR data the extra capacity must not manufacture a
    magnitude effect. Note the target is *not* 1: at rho = 0.6 a homoscedastic
    AR(1) already produces a quartile ratio of 1.61, because conditioning on a
    set of parent values picks up the spread of the conditional mean. The
    control is that the fitted kernel reproduces that null and nothing beyond it.
    """
    grid, _ = make_grid(8.0, 401)
    rng = rng_for("scale-kernel-gauss-control")
    n = 200000
    p = rng.standard_normal(n)
    c = 0.6 * p + np.sqrt(1 - 0.36) * rng.standard_normal(n)
    xi = _pairs_to_xi(grid, p, c)

    k = ScaleMixtureKernel.init(4, rho=0.3, var=1.0, rng=rng)
    for _ in range(25):
        k = k.m_step(xi_stats(xi), grid)

    null = linear_ar_magnitude_ratio(0.6)
    assert abs(k.magnitude_ratio(grid) - null) < 0.08, (
        f"fitted {k.magnitude_ratio(grid):.3f} vs linear-AR null {null:.3f}"
    )
    eff_rho = float((k.gate(grid) * k.rho[None, :]).sum(1).mean())
    assert abs(eff_rho - 0.6) < 0.08


def test_gradient_interface_is_refused_explicitly():
    grid, _ = make_grid(8.0, 201)
    k = ScaleMixtureKernel.init(2, rho=0.4, var=1.0, rng=rng_for("scale-kernel-grad"))
    with pytest.raises(NotImplementedError, match="ECM"):
        k.grad_log_transition_matrix(grid)


# ----------------------------------------------------------------------------
# magnitude_diagnostics
#
# This is the measurement that decides whether the scale-mixture family did what
# it was built for. exp_23 finds an empirical Q4/Q1 excess of 1.86 (HH) to 2.33
# (HL) at the finest scale boundary; a fitted kernel either reproduces that or it
# does not, and held-out likelihood cannot tell the difference -- a family with
# more parameters can win on likelihood while leaving the magnitude structure
# untouched. So the diagnostic itself needs to be right.
# ----------------------------------------------------------------------------

import dataclasses  # noqa: E402

from src.bp_grid import make_grid  # noqa: E402
from src.kernels import GaussianAR1Kernel  # noqa: E402
from src.scale_kernel import magnitude_diagnostics  # noqa: E402


def _flat_scale_mixture(rho, n_components=4, seed=0):
    """A ScaleMixtureKernel degenerated to a plain AR(1).

    Identical components and a dead gate (beta = gamma = 0), so the conditional
    variance cannot depend on the parent. Its magnitude ratio must therefore
    equal the closed-form linear null -- which makes this the one case where the
    general path has an exactly known answer.
    """
    k = ScaleMixtureKernel.init(
        n_components, rho=rho, var=1.0 - rho**2,
        rng=np.random.default_rng(seed),
    )
    return dataclasses.replace(
        k,
        rho=np.full(n_components, rho),
        s2=np.full(n_components, 1.0 - rho**2),
        beta=np.zeros(n_components),
        gamma=np.zeros(n_components),
    )


@pytest.mark.parametrize("rho", [0.15, 0.45, 0.60])
def test_linear_family_excess_is_exactly_one(rho):
    """A linear-AR kernel has no magnitude dependence to find, by construction.

    Runs on the gaussian and mixture arms of every fit, so a convention or
    quadrature error in the diagnostic shows up there before it is used to make
    a claim about the scale mixture.
    """
    grid, _ = make_grid(8.0, 1201)
    out = magnitude_diagnostics(GaussianAR1Kernel(rho=rho, q=1.0 - rho**2), grid)
    assert out["rho_implied"] == pytest.approx(rho)
    assert out["magnitude_excess"] == pytest.approx(1.0, abs=1e-12)


def test_implied_rho_recovers_the_slope_of_a_degenerate_mixture():
    """With every component sharing one rho, the linear slope is that rho."""
    grid, _ = make_grid(8.0, 1201)
    for rho in (0.2, 0.45):
        out = magnitude_diagnostics(_flat_scale_mixture(rho), grid)
        assert out["rho_implied"] == pytest.approx(rho, abs=2e-3)


@pytest.mark.parametrize("m_grid", [1593, 771, 349, 151, 65])
def test_diagnostic_error_is_small_at_every_grid_the_fit_uses(m_grid):
    """Bound the diagnostic's own discretisation error where it will be used.

    The per-depth meshes in the real exp_24 fit are [1593, 771, 349, 151, 65],
    and the quartile bands are hard masks on grid points -- so which points fall
    inside jumps discretely and the error does *not* fall monotonically with M.
    At M=65 only three points lie in the Q1 band. Measured against the degenerate
    kernel's exact answer the error stays under 1% at every one of these sizes,
    which is ~200x smaller than the 1.86-2.33 excess being measured.

    Asserted rather than described, because the temptation on seeing a coarse
    grid is to refine it, and refining does not monotonically help here.
    """
    grid, _ = make_grid(8.0, m_grid)
    out = magnitude_diagnostics(_flat_scale_mixture(0.45), grid)
    assert abs(out["magnitude_excess"] - 1.0) < 1e-2


def test_a_live_gate_produces_excess_above_one():
    """The direction the whole family exists for: gated variance lifts the ratio
    above what any linear-AR kernel with the same slope could produce."""
    grid, _ = make_grid(8.0, 1201)
    k = ScaleMixtureKernel(
        rho=np.array([0.45, 0.45]),
        s2=np.array([0.2, 2.0]),          # components differ in scale
        beta=np.array([0.0, -1.0]),
        gamma=np.array([0.0, 1.0]),       # large |a| favours the wide component
    )
    out = magnitude_diagnostics(k, grid)
    assert out["magnitude_excess"] > 1.05
    assert out["magnitude_ratio"] > out["magnitude_null"]
