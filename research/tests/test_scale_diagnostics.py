"""Mixture scale diagnostics: is the fitted density resolved by the grid it lives on?

`_VAR_FLOOR = 1e-6` permits a component standard deviation of 1e-3 against a default grid
spacing of 0.04 -- forty times narrower than a cell. A high-capacity mixture can raise the
quadrature likelihood with such a spike without it corresponding to any feature of the
innovation law, which is why the capacity sweep's small C=12->16 gain cannot be interpreted
until these numbers are on the record.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.kernels import MixtureInnovationKernel


def _kernel(s2, pi=None, mu=None):
    s2 = np.asarray(s2, dtype=float)
    c = len(s2)
    pi = np.full(c, 1.0 / c) if pi is None else np.asarray(pi, dtype=float)
    mu = np.zeros(c) if mu is None else np.asarray(mu, dtype=float)
    return MixtureInnovationKernel(rho=0.85, pi=pi, mu=mu, s2=s2)


def test_flags_an_underresolved_component():
    """A component at the variance floor must be reported unresolved, not silently used."""
    grid, _ = make_grid(8.0, 401)          # h = 0.04
    d = _kernel([1e-6, 0.25]).scale_diagnostics(grid)

    assert d["grid_spacing"] == pytest.approx(0.04)
    assert d["s_min"] == pytest.approx(1e-3)
    assert d["s_min_over_h"] == pytest.approx(0.025, rel=1e-6)
    assert d["resolved"] is False


def test_accepts_a_resolved_mixture():
    grid, _ = make_grid(8.0, 401)
    d = _kernel([0.04, 0.25]).scale_diagnostics(grid)   # s_min = 0.2 = 5h
    assert d["s_min_over_h"] == pytest.approx(5.0, rel=1e-6)
    assert d["resolved"] is True


def test_effective_component_count_sees_collapse():
    """A nominally large mixture that has collapsed onto two live components must report
    two, since that is what distinguishes real capacity from padding."""
    grid, _ = make_grid(8.0, 401)
    live = _kernel([0.1] * 8, pi=[0.5, 0.5] + [1e-12] * 6)
    assert live.scale_diagnostics(grid)["effective_n_components"] == pytest.approx(2.0, abs=1e-6)

    even = _kernel([0.1] * 8)
    assert even.scale_diagnostics(grid)["effective_n_components"] == pytest.approx(8.0, abs=1e-9)


def test_column_mass_residual_grows_when_components_are_narrow():
    """The mechanism, not just the flag.

    An under-resolved component breaks the quadrature: its mass is not captured by the
    trapezoid rule, so transition columns stop integrating to one. This asserts the residual
    is orders of magnitude worse for the spike than for the resolved mixture, which is what
    makes `resolved` a statement about the numerics rather than an arbitrary threshold.
    """
    grid, _ = make_grid(8.0, 401)
    spike = _kernel([1e-6, 0.25]).scale_diagnostics(grid)
    fine = _kernel([0.04, 0.25]).scale_diagnostics(grid)
    assert spike["column_mass_residual_interior"] > 100 * fine["column_mass_residual_interior"]
