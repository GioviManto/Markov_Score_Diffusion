"""Pins the lattice bias of the Laplace M-step's rho.

Remark `rem:quantization` in report/em_bp_learning.tex, and the corresponding
sentence at paper/main.tex:496, rest on a claim that nothing in the suite
currently checks: the exact maximiser of Q for the Laplace kernel is a
*lattice-valued* estimator, so its rho carries a bias that grid refinement does
not remove. That claim is load-bearing -- it is the stated reason the rate
experiments use the smooth kernels instead -- and if a future change to
`_weighted_median` or to `make_grid` quietly fixed or worsened it, the prose
would silently stop matching the code.

The mechanism: Q(rho) reduces to a weighted mean absolute residual, whose
minimiser is a breakpoint, i.e. one of the ratios u_k/u_j. `make_grid` is a
uniform grid through the origin, so grid[j] = j'*dx for integer j' and every
ratio is a rational a/b with |a|,|b| <= (M-1)/2. Low-denominator values pool the
aliases of many pairs (4/5 collects 8/10, 12/15, ...) and become attractors.

This is a property of the estimator on a symmetric uniform grid, not of EM, and
not of the mixture kernel -- whose rho block solves continuous weighted least
squares and is therefore not lattice-valued. `test_mixture_rho_is_not_lattice_valued`
pins that contrast, since it is what keeps the paper's headline recovery numbers
(which come from the mixture kernel) clear of this caveat.

Run:  python -m pytest tests/test_laplace_rho_lattice.py -q
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from src.bp_grid import make_grid
from src.em import e_step_multi
from src.kernels import LaplaceAR1Kernel, MixtureInnovationKernel
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.utils import rng_for

N_SITES = 16
T_TRAIN = (0.2, 0.8)


def _groups(prior, n_chains, rng, t_values=T_TRAIN):
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    out = []
    for t in t_values:
        alpha, delta = alpha_delta(t)
        out.append((alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape),
                    alpha, delta))
    return out


def _fit_rho(kernel, grid, weights, groups, n_iters=12):
    current = kernel
    for _ in range(n_iters):
        stats = e_step_multi(
            grid, weights, current.log_transition_matrix(grid), groups
        )
        current = current.m_step(stats, grid)
    return current


def test_laplace_rho_lands_on_a_low_order_rational():
    """Every fitted rho is exactly a ratio of two grid indices."""
    grid, weights = make_grid(8.0, 201)
    rng = rng_for("test-lattice-rational")
    groups = _groups(LaplaceAR1(0.8), 48, rng)

    fitted = _fit_rho(LaplaceAR1Kernel(0.4, 0.7), grid, weights, groups)

    half = (len(grid) - 1) // 2
    frac = Fraction(fitted.rho).limit_denominator(half)
    assert abs(float(frac) - fitted.rho) < 1e-9, (
        f"rho={fitted.rho!r} is not a ratio of grid indices; the M-step is no "
        "longer lattice-valued and rem:quantization needs rewriting"
    )
    assert frac.denominator <= half


def test_grid_refinement_does_not_remove_the_bias_at_an_off_lattice_truth():
    """The headline of rem:quantization: 0.7913 stays pinned at 4/5.

    The committed sweep (outputs/.../laplace_quantization.csv) reports rho_hat =
    0.800000 at M = 201, 401, 801 and 1601 with a constant 0.0087 bias. A cheaper
    configuration is used here, so the assertion is the qualitative one: the
    error must not shrink the way a genuine discretisation error would.
    """
    rho_star = 0.7913
    rng = rng_for("test-lattice-refine")
    prior = LaplaceAR1(rho_star)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(48)])

    errs = {}
    for M in (201, 401, 801):
        grid, weights = make_grid(8.0, M)
        groups = []
        g_rng = rng_for("test-lattice-refine-noise")
        for t in T_TRAIN:
            alpha, delta = alpha_delta(t)
            groups.append((alpha * A + np.sqrt(delta) * g_rng.standard_normal(A.shape),
                           alpha, delta))
        fitted = _fit_rho(LaplaceAR1Kernel(0.4, 0.7), grid, weights, groups)
        errs[M] = abs(fitted.rho - rho_star)

    # A real O(dx) error would fall by ~4x across a 4x refinement. This one does
    # not fall at all -- it is a bias, and that is the whole point of the remark.
    assert errs[801] > 0.25 * errs[201], (
        f"the lattice bias now decays under refinement ({errs}); "
        "rem:quantization claims it does not"
    )


def test_mixture_rho_is_not_lattice_valued():
    """The contrast that protects the paper's headline recovery numbers.

    The mixture rho block is a weighted least-squares solve, so its output varies
    continuously and generically is *not* a ratio of grid indices. exp_18's
    reported [0.8517, 0.8520] against a truth of 0.85 is therefore a genuine
    estimate, not a lattice point.
    """
    grid, weights = make_grid(8.0, 201)
    rng = rng_for("test-lattice-mixture")
    groups = _groups(LaplaceAR1(0.85), 48, rng)

    start = MixtureInnovationKernel.init(
        4, rho=0.3, var=0.8, rng=rng_for("test-lattice-mixture-init")
    )
    fitted = _fit_rho(start, grid, weights, groups)

    half = (len(grid) - 1) // 2
    frac = Fraction(fitted.rho).limit_denominator(half)
    assert abs(float(frac) - fitted.rho) > 1e-12, (
        f"mixture rho={fitted.rho!r} landed exactly on the ratio lattice; the "
        "rho block is supposed to be a continuous weighted least-squares solve"
    )
