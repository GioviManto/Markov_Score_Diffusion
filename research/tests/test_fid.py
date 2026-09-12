"""IS and FID, against closed forms and against an independent reference.

Everything here is numpy: the network is lazily imported inside
`inception_activations` and nothing in this file touches it, so the statistical
half of the metric stays testable on a machine with no torch. That is the half
that can be checked against a closed form, and therefore the half where a silent
error would otherwise survive.

The FID checks deliberately avoid re-using `scipy.linalg.sqrtm` on the *product*
`S1 S2`, which is what the implementation does. For the general case the
reference here computes `Tr((S1 S2)^{1/2})` from the eigenvalues of the symmetric
`S1^{1/2} S2 S1^{1/2}` instead -- same quantity, different route, so a mistake in
the implementation's handling of the non-symmetric product cannot cancel.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from src.fid import (
    frechet_decomposition, predicted_mean_term,
    ActivationStats, bias_curve, blur_monotonicity, fid_from_samples,
    fid_from_stats, frechet_distance, inception_score, real_vs_real_floor,
)


def _reference_frechet(mu1, s1, mu2, s2) -> float:
    """Tr((S1 S2)^{1/2}) via the symmetric conjugate, which has real eigenvalues."""
    root1 = linalg.sqrtm(s1)
    assert not np.iscomplexobj(root1) or np.max(np.abs(root1.imag)) < 1e-10
    root1 = np.real(root1)
    middle = root1 @ s2 @ root1
    evals = np.linalg.eigvalsh(middle)
    tr_cross = float(np.sum(np.sqrt(np.maximum(evals, 0.0))))
    diff = np.asarray(mu1) - np.asarray(mu2)
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * tr_cross)


def _psd(dim, rng, scale=1.0):
    a = rng.standard_normal((dim, dim + 4))
    return scale * (a @ a.T) / (dim + 4)


# ---------------------------------------------------------------- Frechet ---


def test_identical_gaussians_have_zero_distance():
    rng = np.random.default_rng(0)
    mu, sigma = rng.standard_normal(6), _psd(6, rng)
    assert abs(frechet_distance(mu, sigma, mu, sigma)) < 1e-8


def test_scalar_case_matches_the_closed_form():
    """In 1-D, d^2 = (m1 - m2)^2 + (s1 - s2)^2 exactly."""
    for m1, m2, s1, s2 in [(0.0, 1.0, 1.0, 1.0), (-2.0, 3.0, 0.5, 2.0)]:
        got = frechet_distance([m1], [[s1**2]], [m2], [[s2**2]])
        want = (m1 - m2) ** 2 + (s1 - s2) ** 2
        assert abs(got - want) < 1e-9, f"{got} != {want}"


def test_diagonal_case_matches_the_closed_form():
    """Commuting covariances reduce to a sum of the 1-D case per coordinate."""
    rng = np.random.default_rng(1)
    d1, d2 = rng.uniform(0.2, 3.0, 8), rng.uniform(0.2, 3.0, 8)
    mu1, mu2 = rng.standard_normal(8), rng.standard_normal(8)
    got = frechet_distance(mu1, np.diag(d1), mu2, np.diag(d2))
    want = float(np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(d1) - np.sqrt(d2)) ** 2))
    assert abs(got - want) < 1e-8


def test_general_case_matches_an_independent_reference():
    rng = np.random.default_rng(2)
    for _ in range(5):
        mu1, mu2 = rng.standard_normal(10), rng.standard_normal(10)
        s1, s2 = _psd(10, rng), _psd(10, rng, scale=2.5)
        got = frechet_distance(mu1, s1, mu2, s2)
        want = _reference_frechet(mu1, s1, mu2, s2)
        assert abs(got - want) < 1e-7 * max(1.0, abs(want))


def test_distance_is_symmetric():
    rng = np.random.default_rng(3)
    mu1, mu2 = rng.standard_normal(7), rng.standard_normal(7)
    s1, s2 = _psd(7, rng), _psd(7, rng, scale=0.4)
    ab = frechet_distance(mu1, s1, mu2, s2)
    ba = frechet_distance(mu2, s2, mu1, s1)
    assert abs(ab - ba) < 1e-8 * max(1.0, abs(ab))


def test_a_pure_translation_adds_its_squared_norm():
    rng = np.random.default_rng(4)
    mu, sigma = rng.standard_normal(5), _psd(5, rng)
    shift = np.array([0.3, -1.2, 0.0, 2.0, 0.5])
    got = frechet_distance(mu, sigma, mu + shift, sigma)
    assert abs(got - float(shift @ shift)) < 1e-8


def test_dimension_mismatch_is_refused():
    with pytest.raises(ValueError, match="same dimension"):
        frechet_distance(np.zeros(3), np.eye(3), np.zeros(4), np.eye(4))


# ------------------------------------------------------------ the guards ---


def test_mismatched_sample_sizes_are_refused():
    """The comparison the module exists to prevent."""
    rng = np.random.default_rng(5)
    a = ActivationStats.from_activations(rng.standard_normal((500, 4)), "w")
    b = ActivationStats.from_activations(rng.standard_normal((900, 4)), "w")
    with pytest.raises(ValueError, match="biased downwards in n"):
        fid_from_stats(a, b)


def test_mismatched_inception_weights_are_refused():
    rng = np.random.default_rng(6)
    a = ActivationStats.from_activations(rng.standard_normal((300, 4)), "torchvision")
    b = ActivationStats.from_activations(rng.standard_normal((300, 4)), "tf-slim")
    with pytest.raises(ValueError, match="different networks"):
        fid_from_stats(a, b)


def test_singular_covariance_is_refused_by_default():
    """n <= d makes the sample covariance rank-deficient, not merely noisy."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((40, 64))
    b = rng.standard_normal((40, 64))
    with pytest.raises(ValueError, match="rank-deficient"):
        fid_from_samples(a, b)
    fid_from_samples(a, b, allow_small_n=True)      # explicit opt-out works


# ------------------------------------------------------------- behaviour ---


def test_fid_grows_with_a_real_distributional_difference():
    rng = np.random.default_rng(8)
    real = rng.standard_normal((4000, 8))
    near = rng.standard_normal((4000, 8)) * 1.05
    far = rng.standard_normal((4000, 8)) * 2.0 + 1.5
    d_near = fid_from_samples(real, near, allow_small_n=True)
    d_far = fid_from_samples(real, far, allow_small_n=True)
    assert 0.0 <= d_near < d_far


def test_real_vs_real_floor_is_small_but_not_zero():
    """The noise floor. Zero would mean the estimator had no variance."""
    rng = np.random.default_rng(9)
    real = rng.standard_normal((4000, 8))
    out = real_vs_real_floor(real, n_repeats=5, allow_small_n=True)
    assert out["floor_mean"] > 0.0
    assert out["floor_mean"] < 0.2
    assert out["n_per_half"] == 2000


def test_the_floor_is_the_resolution_of_a_model_comparison():
    """A 'difference' below the floor is not a difference.

    This is the property the floor is computed for, asserted rather than
    described: a model whose samples are drawn from the reference distribution
    itself scores within the floor, so any claim resting on a gap that small is
    unsupported.
    """
    rng = np.random.default_rng(10)
    real = rng.standard_normal((6000, 8))
    impostor = rng.standard_normal((3000, 8))       # same law, different draw
    floor = real_vs_real_floor(real, n_repeats=5, allow_small_n=True)
    d = fid_from_samples(real[:3000], impostor, allow_small_n=True)
    assert d < floor["floor_mean"] + 5.0 * max(floor["floor_std"], 1e-9) + 0.05


def test_bias_curve_falls_with_n():
    """FID is a plug-in estimator: it decreases in n even for identical laws."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal((8000, 8))
    b = rng.standard_normal((8000, 8))
    rows = bias_curve(a, b, sizes=[200, 800, 3200], n_repeats=4, allow_small_n=True)
    vals = [r["fid_mean"] for r in rows]
    assert len(vals) == 3
    assert vals[0] > vals[1] > vals[2], f"bias curve is not falling: {vals}"


def test_bias_curve_skips_sizes_larger_than_the_sample():
    rng = np.random.default_rng(12)
    a = rng.standard_normal((500, 4))
    rows = bias_curve(a, a, sizes=[100, 400, 9000], n_repeats=2, allow_small_n=True)
    assert [r["n"] for r in rows] == [100, 400]


# ------------------------------------------------------------- blur check ---


def test_blur_monotonicity_accepts_a_rising_sequence():
    out = blur_monotonicity({0.0: 2.0, 0.5: 8.0, 1.0: 25.0, 2.0: 60.0})
    assert out["monotone"]
    assert out["violations"] == []
    assert out["sigmas"] == [0.0, 0.5, 1.0, 2.0]


def test_blur_monotonicity_reports_where_it_broke():
    out = blur_monotonicity({0.0: 2.0, 0.5: 30.0, 1.0: 12.0})
    assert not out["monotone"]
    assert len(out["violations"]) == 1
    assert out["violations"][0]["from_sigma"] == 0.5
    assert out["violations"][0]["to_sigma"] == 1.0


# ------------------------------------------------------------------- IS ----


def test_inception_score_of_confident_and_uniform_predictions_is_the_class_count():
    """One-hot predictions spread evenly over C classes score exactly C.

    `n_splits=1` on purpose. The identity is exact only when p(y) is *exactly*
    uniform, and p(y) is estimated within each split: at n_splits=5 the random
    partition leaves each split's class counts multinomial rather than balanced,
    which lowers H(p(y)) and gives ~7.95 for C=10. That is the estimator behaving
    correctly, not an error -- but it makes the closed form inexact, so the split
    that admits a closed form is the one to assert on.
    """
    n_classes, per_class = 10, 50
    probs = np.repeat(np.eye(n_classes), per_class, axis=0)
    mean, _ = inception_score(probs, n_splits=1)
    assert abs(mean - n_classes) < 1e-6


def test_inception_score_falls_when_splits_unbalance_the_marginal():
    """The flip side, asserted so the split count is never treated as cosmetic.

    Only the direction is asserted. The size of the drop depends on the random
    partition and on how many samples each split gets, so pinning a window would
    be pinning this seed rather than the property -- and the property is that two
    IS numbers computed with different `n_splits` are not comparable.
    """
    probs = np.repeat(np.eye(10), 50, axis=0)
    one, one_std = inception_score(probs, n_splits=1)
    many, many_std = inception_score(probs, n_splits=5)
    assert many < one
    assert one_std == 0.0, "a single split has no across-split spread to report"
    assert many_std > 0.0


def test_inception_score_of_identical_predictions_is_one():
    """No marginal diversity: KL is zero for every image."""
    probs = np.tile(np.array([0.7, 0.2, 0.1]), (300, 1))
    mean, std = inception_score(probs, n_splits=5)
    assert abs(mean - 1.0) < 1e-9
    assert std < 1e-9


def test_inception_score_is_blind_to_within_class_diversity():
    """The documented weakness, asserted so it cannot be forgotten in reporting.

    A sample of one image per class and a sample of many varied images per class
    are indistinguishable to IS, because it only ever sees p(y|x).
    """
    n_classes = 8
    one_each = np.eye(n_classes)
    many_each = np.repeat(np.eye(n_classes), 40, axis=0)
    a, _ = inception_score(one_each, n_splits=1)
    b, _ = inception_score(many_each, n_splits=1)
    assert abs(a - b) < 1e-6
    assert abs(a - n_classes) < 1e-6


def test_inception_score_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, n_classes\)"):
        inception_score(np.zeros(10))


# ------------------------------------------------------- the blur reference ---
# `exp_30_fid_validation` reads the blur curve as a pass/fail gate on the whole
# FID pipeline, which only works if the blur itself is right. A blur that is
# weaker than requested still produces a rising curve, so the gate would pass
# while the sigmas on the axis meant something other than what they say.

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp_30_fid_validation import _gaussian_blur, _to_unit  # noqa: E402


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 4.0])
def test_blur_matches_scipy_gaussian_filter(sigma):
    """Machine precision against ndimage, including at the boundary.

    The boundary is the whole point of this test. numpy's `reflect` drops the
    edge sample and scipy.ndimage's repeats it -- the same word for different
    padding -- and the first version of `_gaussian_blur` used numpy's, which
    disagreed by up to 0.13 on [0, 1] data. Interior pixels were exact, so an
    interior-only check would have passed. Inception sees the edges.
    """
    from scipy import ndimage
    rng = np.random.default_rng(0)
    imgs = rng.random((3, 32, 32))
    got = _gaussian_blur(imgs, sigma)
    want = np.stack([
        ndimage.gaussian_filter(im, sigma, mode="reflect", truncate=4.0)
        for im in imgs
    ])
    assert np.max(np.abs(got - want)) < 1e-12


def test_blur_of_zero_sigma_is_the_identity():
    rng = np.random.default_rng(1)
    imgs = rng.random((2, 16, 16))
    assert np.array_equal(_gaussian_blur(imgs, 0.0), imgs)


def test_blur_destroys_variance_monotonically():
    """The property the FID curve is supposed to detect, checked directly."""
    rng = np.random.default_rng(2)
    imgs = rng.random((4, 32, 32))
    variances = [float(_gaussian_blur(imgs, s).var()) for s in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(variances, variances[1:])), variances


def test_blur_preserves_the_mean():
    """A normalised kernel with symmetric padding does not shift the level."""
    rng = np.random.default_rng(3)
    imgs = rng.random((3, 24, 24))
    for sigma in (0.5, 2.0):
        assert abs(_gaussian_blur(imgs, sigma).mean() - imgs.mean()) < 1e-10


def test_to_unit_maps_into_the_range_inception_requires():
    """`inception_activations` refuses anything outside [0, 1] rather than
    rescaling, so the conversion has to land inside it for real CIFAR ranges."""
    rng = np.random.default_rng(4)
    raw = rng.standard_normal((5, 8, 8)) * 40.0 - 7.0
    unit = _to_unit(raw)
    assert unit.min() >= 0.0 and unit.max() <= 1.0
    assert abs(unit.min()) < 1e-12 and abs(unit.max() - 1.0) < 1e-12


def test_bias_curve_flags_the_degenerate_largest_size():
    """At n == len(sample) there is no resampling, so the spread is not a number.

    The real run made this concrete: at n=10000 out of 10000 available, four
    "repeats" drew the same permutation and reported std 0.0000 -- which reads
    as a perfectly stable estimate and is really no resampling at all. It is
    also the largest-n row, the one most likely to be quoted.
    """
    rng = np.random.default_rng(20)
    a = rng.standard_normal((1000, 6))
    b = rng.standard_normal((1000, 6))
    rows = bias_curve(a, b, sizes=[250, 1000], n_repeats=4, allow_small_n=True)

    small, full = rows[0], rows[1]
    assert small["resampled"] is True
    assert small["n_repeats"] == 4
    assert small["fid_std"] > 0.0

    assert full["resampled"] is False
    assert full["n_repeats"] == 1
    assert np.isnan(full["fid_std"]), "a non-resampled row must not report 0.0"


# ------------------------------------------------- the bias decomposition ---


def test_decomposition_sums_to_the_total():
    rng = np.random.default_rng(30)
    mu1, mu2 = rng.standard_normal(9), rng.standard_normal(9)
    s1, s2 = _psd(9, rng), _psd(9, rng, scale=1.7)
    out = frechet_decomposition(mu1, s1, mu2, s2)
    assert out["mean_term"] + out["covariance_term"] == pytest.approx(out["fid"])
    assert out["mean_term"] == pytest.approx(float((mu1 - mu2) @ (mu1 - mu2)))


def test_identical_covariances_leave_only_the_mean_term():
    rng = np.random.default_rng(31)
    mu1, mu2 = rng.standard_normal(6), rng.standard_normal(6)
    sigma = _psd(6, rng)
    out = frechet_decomposition(mu1, sigma, mu2, sigma)
    assert abs(out["covariance_term"]) < 1e-8
    assert out["mean_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_equal_means_leave_only_the_covariance_term():
    rng = np.random.default_rng(32)
    mu = rng.standard_normal(6)
    out = frechet_decomposition(mu, _psd(6, rng), mu, _psd(6, rng, scale=3.0))
    assert out["mean_term"] == pytest.approx(0.0)
    assert out["covariance_term"] > 0.0


def test_predicted_mean_term_matches_simulation():
    """`2 Tr(Sigma) / n` is exact, so simulation must land on it, not near it.

    Checked on a *non-Gaussian* sample as well, because the identity needs only a
    finite covariance -- if this were quietly relying on Gaussianity the
    extrapolation it supports would be resting on an assumption nobody stated.
    """
    rng = np.random.default_rng(33)
    dim, n, trials = 12, 400, 4000
    scale = rng.uniform(0.5, 2.0, dim)

    for label, draw in (
        ("gaussian", lambda m: rng.standard_normal((m, dim)) * scale),
        ("exponential", lambda m: (rng.exponential(1.0, (m, dim)) - 1.0) * scale),
    ):
        pop_cov_trace = float(np.sum(scale**2))    # unit-variance base in both
        want = predicted_mean_term(np.diag(scale**2), n)
        assert want == pytest.approx(2.0 * pop_cov_trace / n)

        got = np.mean([
            float(np.sum((draw(n).mean(0) - draw(n).mean(0)) ** 2))
            for _ in range(trials // 20)
        ])
        assert got == pytest.approx(want, rel=0.25), f"{label}: {got} vs {want}"


def test_mean_term_falls_as_one_over_n():
    rng = np.random.default_rng(34)
    sigma = np.diag(rng.uniform(0.5, 2.0, 10))
    a = predicted_mean_term(sigma, 1000)
    b = predicted_mean_term(sigma, 2000)
    assert a / b == pytest.approx(2.0)
