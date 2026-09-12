"""The exact identity  s_hat - s_ref = (alpha/Delta)(m_hat - m_ref)."""

import numpy as np

from src.bp_grid import score_from_posterior_mean
from src.metrics import score_mean_identity_residual
from src.noising import alpha_delta
from src.utils import rng_for


def test_identity_holds_for_arbitrary_means():
    rng = rng_for("test-identity")
    n = 30
    x = rng.standard_normal(n)
    m_ref = rng.standard_normal(n)
    m_hat = m_ref + 0.3 * rng.standard_normal(n)
    for t in (0.05, 0.4, 2.0):
        alpha, delta = alpha_delta(t)
        s_ref = score_from_posterior_mean(x, m_ref, alpha, delta)
        s_hat = score_from_posterior_mean(x, m_hat, alpha, delta)
        res = score_mean_identity_residual(s_hat, s_ref, m_hat, m_ref, alpha, delta)
        assert res < 1e-12


def test_amplification_factor_grows_at_small_t():
    # alpha/Delta ~ 1/(2t): the same mean error costs more score error at small t.
    a1, d1 = alpha_delta(0.05)
    a2, d2 = alpha_delta(1.0)
    assert a1 / d1 > 5 * (a2 / d2)
