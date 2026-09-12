"""Validation of src/sample_metrics.py.

The load-bearing tests here are the ones that check a metric can *detect the thing it
exists to detect*. A distributional metric that returns a plausible number on every input
is worse than none, because it will be reported.

Run:  python -m pytest tests/test_sample_metrics.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.priors import LaplaceAR1
from src.sample_metrics import (
    bayes_risk,
    bootstrap_se,
    compare_distributions,
    covariance_error,
    excess_kurtosis,
    histogram_kl,
    innovations,
    pointwise_ladder,
)
from src.utils import rng_for


# ---------------------------------------------------------------------------
# Innovation recovery
# ---------------------------------------------------------------------------

def test_innovations_invert_the_chain_construction():
    """Building a chain from known innovations and recovering them must round-trip."""
    rng = np.random.default_rng(0)
    rho, n, m = 0.85, 24, 500
    e = rng.laplace(0.0, 1.0, (m, n - 1))
    a = np.zeros((m, n))
    a[:, 0] = rng.normal(size=m)
    for i in range(1, n):
        a[:, i] = rho * a[:, i - 1] + e[:, i - 1]

    rec = innovations(a, rho)
    assert np.max(np.abs(rec - e)) < 1e-10


def test_innovations_rejects_wrong_shape():
    with pytest.raises(ValueError):
        innovations(np.zeros(10), 0.5)


# ---------------------------------------------------------------------------
# Kurtosis: the statistic the whole non-Gaussian story rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sampler,truth,tol",
    [
        (lambda r, n: r.normal(0.0, 1.0, n), 0.0, 0.15),
        (lambda r, n: r.laplace(0.0, 1.0, n), 3.0, 0.9),
        (lambda r, n: r.uniform(-1.0, 1.0, n), -1.2, 0.06),
    ],
)
def test_excess_kurtosis_recovers_known_families(sampler, truth, tol):
    """The three families whose true excess kurtosis the project quotes.

    Tolerances differ by family on purpose: the Laplace fourth moment is genuinely
    high-variance (the compendium records a kurtosis estimate reading 2.91, 3.45, 3.70,
    3.29, 2.64 across replicates), so demanding tight recovery here would be demanding
    something the estimator cannot deliver.
    """
    rng = np.random.default_rng(1)
    x = sampler(rng, 200_000)
    assert abs(excess_kurtosis(x) - truth) < tol


def test_excess_kurtosis_degenerate_inputs_are_nan_not_zero():
    """A constant array has no defined kurtosis; returning 0.0 would read as 'Gaussian'."""
    assert np.isnan(excess_kurtosis(np.full(100, 2.5)))
    assert np.isnan(excess_kurtosis(np.array([1.0, 2.0])))


def test_bootstrap_se_resamples_chains_not_entries():
    """SE must reflect chain-level resampling, and must shrink like 1/sqrt(n_chains)."""
    rng = np.random.default_rng(2)
    small = rng.laplace(0.0, 1.0, (250, 30))
    large = rng.laplace(0.0, 1.0, (4000, 30))
    se_small = bootstrap_se(small, excess_kurtosis, n_boot=150, seed=0)
    se_large = bootstrap_se(large, excess_kurtosis, n_boot=150, seed=0)
    assert se_large < se_small
    # 16x more chains should buy roughly 4x, allowing generous slack for a 4th moment.
    assert 1.7 < se_small / se_large < 9.0


# ---------------------------------------------------------------------------
# The decisive test: does the bundle separate the failure it was built to catch?
# ---------------------------------------------------------------------------

def test_gaussian_generation_is_flagged_as_washing_out_heavy_tails():
    """The exp_05 failure mode, reproduced synthetically.

    A model that matches the covariance but Gaussianises the innovations is *exactly* what
    the Gaussian closure does (measured excess kurtosis 0.12 against a true 2.7-2.9). The
    covariance metrics must stay clean while the kurtosis metric fires -- if both fire, the
    bundle cannot attribute the failure; if neither does, it is useless.
    """
    rng = np.random.default_rng(3)
    rho, n, m = 0.85, 32, 4000
    q = 1.0 - rho ** 2
    sigma = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    def build(e):
        a = np.zeros((m, n))
        a[:, 0] = rng.normal(0.0, 1.0, m)
        for i in range(1, n):
            a[:, i] = rho * a[:, i - 1] + e[:, i - 1]
        return a

    b = np.sqrt(q / 2.0)
    a_true = build(rng.laplace(0.0, b, (m, n - 1)))
    a_gauss = build(rng.normal(0.0, np.sqrt(q), (m, n - 1)))

    good = compare_distributions(a_true, a_true, rho, sigma, 3.0, q, name="true", seed=0)
    bad = compare_distributions(a_gauss, a_true, rho, sigma, 3.0, q, name="gaussian", seed=0)

    # The kurtosis metric must separate them decisively.
    assert good.kurtosis_gap_in_se() < 3.0, "true sample should sit near its own truth"
    assert bad.kurtosis_gap_in_se() > 8.0, "Gaussianised innovations must be flagged"

    # ...while the covariance metric must NOT, since both share rho^|i-j| by construction.
    assert bad.cov_worst_lag_abs < 0.10
    assert abs(good.cov_worst_lag_abs - bad.cov_worst_lag_abs) < 0.08

    # And the KL should rank them in the same direction as the kurtosis.
    assert bad.innov_kl > good.innov_kl


def test_small_sample_is_annotated_rather_than_silently_reported():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(120, 16))
    sigma = np.eye(16)
    c = compare_distributions(a, a, 0.0, sigma, 0.0, 1.0, seed=0)
    assert any("generated chains" in s for s in c.notes)


def test_non_finite_generated_sample_is_reported():
    """A blown-up integrator must be named, not averaged over."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(600, 16))
    a[3, 4] = np.inf
    sigma = np.eye(16)
    c = compare_distributions(a, a, 0.0, sigma, 0.0, 1.0, seed=0)
    assert any("non-finite" in s for s in c.notes)


# ---------------------------------------------------------------------------
# Covariance and KL
# ---------------------------------------------------------------------------

def test_covariance_error_is_zero_against_itself():
    rng = np.random.default_rng(6)
    n = 20
    sigma = 0.8 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    a = rng.multivariate_normal(np.zeros(n), sigma, size=40_000)
    fro, worst = covariance_error(a, sigma)
    assert fro < 0.05
    assert worst < 0.03


def test_histogram_kl_is_near_zero_for_identical_laws_and_large_for_different():
    rng = np.random.default_rng(7)
    p = rng.normal(size=40_000)
    q_same = rng.normal(size=40_000)
    q_diff = rng.laplace(0.0, 1.0, 40_000)
    assert histogram_kl(p, q_same) < 0.02
    assert histogram_kl(p, q_diff) > histogram_kl(p, q_same)


# ---------------------------------------------------------------------------
# The pointwise ladder and the Tweedie reweighting
# ---------------------------------------------------------------------------

def test_score_reweighting_matches_the_tweedie_factor():
    """The claim that score error is posterior-mean error times alpha/delta.

    This is the reason the report de-emphasises relative score error, so it must be checked
    rather than asserted. The factor is verified directly against its definition, and the
    implied absolute score error against an independent computation.
    """
    rng = np.random.default_rng(8)
    m_star = rng.normal(size=(200, 16))
    m_hat = m_star + 0.01 * rng.normal(size=(200, 16))
    a = m_star + 0.3 * rng.normal(size=(200, 16))

    for t in (0.08, 0.5, 2.4):
        alpha = float(np.exp(-t))
        delta = float(1.0 - np.exp(-2.0 * t))
        d = pointwise_ladder(m_hat, m_star, a, alpha, delta)

        assert abs(d["score_reweighting"] - alpha / delta) < 1e-12
        expected = (alpha / delta) * float(np.linalg.norm(m_hat - m_star))
        assert abs(d["abs_score_error_implied"] - expected) < 1e-9


def test_reweighting_spans_the_claimed_range_across_the_schedule():
    """The compendium claims alpha/delta runs ~6.0 to ~0.09 over t in [0.08, 2.4]."""
    def f(t):
        return np.exp(-t) / (1.0 - np.exp(-2.0 * t))

    assert 5.5 < f(0.08) < 6.5
    assert 0.08 < f(2.4) < 0.10
    assert f(0.08) / f(2.4) > 50.0


def test_denoising_risk_floors_at_bayes_risk():
    """The perfect denoiser attains the floor exactly, and captures 100% of the reduction."""
    rng = np.random.default_rng(9)
    m_star = rng.normal(size=(400, 12))
    a = m_star + 0.5 * rng.normal(size=(400, 12))

    d = pointwise_ladder(m_star, m_star, a, np.exp(-0.5), 1.0 - np.exp(-1.0))
    assert d["mse_vs_bayes_denoiser"] == pytest.approx(0.0, abs=1e-12)
    assert d["denoising_risk"] == pytest.approx(d["bayes_risk_floor"], rel=1e-12)
    assert d["fraction_achievable_captured"] == pytest.approx(1.0, rel=1e-9)


def test_trivial_predictor_captures_none_of_the_achievable_reduction():
    rng = np.random.default_rng(10)
    m_star = rng.normal(size=(400, 12))
    a = m_star + 0.5 * rng.normal(size=(400, 12))
    d = pointwise_ladder(np.zeros_like(m_star), m_star, a, 0.6, 0.6)
    assert abs(d["fraction_achievable_captured"]) < 0.05


def test_bayes_risk_is_not_zero_for_a_genuine_posterior():
    """Guards the mistake that produced an unsatisfiable test criterion earlier."""
    rng = np.random.default_rng(11)
    m_star = rng.normal(size=(300, 10))
    a = m_star + 0.4 * rng.normal(size=(300, 10))
    assert bayes_risk(m_star, a) > 0.1


# ---------------------------------------------------------------------------
# AR-filtered residuals are not innovations at t_min > 0
# ---------------------------------------------------------------------------

def test_ar_residuals_are_correlated_at_positive_t_and_match_the_closed_form():
    """The external review's point, turned into a measurement.

    `r_i = x_i - rho x_{i-1}` applied to a sample of p_{t_min} is not an innovation
    sequence: r_i = alpha eps_i + sqrt(Delta)(z_i - rho z_{i-1}), so adjacent residuals
    carry Cov = -rho Delta. This test confirms the measured lag-1 correlation matches the
    closed form, and -- the part that matters for the paper -- that it is clearly nonzero,
    so calling these innovations overstates what has been recovered.

    The clean chain is the control: there the same filter DOES return innovations, and the
    correlation must vanish.
    """
    from src.sample_metrics import (
        ar_residuals,
        predicted_residual_autocorr,
        residual_autocorrelation,
    )

    rho, t_min = 0.85, 0.02
    prior = LaplaceAR1(rho)
    rng = rng_for("test-ar-residuals")
    a = np.stack([prior.sample(rng, 64) for _ in range(4000)])

    # Clean chain: the filter really does recover innovations.
    clean_acf = residual_autocorrelation(ar_residuals(a, rho), max_lag=2)
    assert abs(clean_acf[0]) < 0.02

    # Noised to t_min, as every generated sample in this project is.
    alpha = float(np.exp(-t_min))
    delta = float(1.0 - np.exp(-2.0 * t_min))
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    measured = residual_autocorrelation(ar_residuals(x, rho), max_lag=2)
    predicted = predicted_residual_autocorr(rho, prior.q, t_min)

    assert measured[0] == pytest.approx(predicted, abs=0.01)
    assert abs(measured[0]) > 0.02, "the correction is only worth making if the effect is real"
    assert predicted < 0.0


def test_innovations_alias_still_works():
    """Deprecated alias must not break committed callers."""
    from src.sample_metrics import ar_residuals, innovations

    a = np.random.default_rng(0).standard_normal((5, 9))
    np.testing.assert_array_equal(innovations(a, 0.8), ar_residuals(a, 0.8))
