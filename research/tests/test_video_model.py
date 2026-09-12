"""The video model: data construction, the exact LL treatment, and fitting.

The load-bearing checks are the two that would let a wrong result look right:
the LL sequence solve against an independent dense computation, and the recovery
of a *known* temporal correlation from noisy observations. Everything else in
the video pipeline sits on top of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.ndimage import gaussian_filter

from src.kernels import GaussianAR1Kernel
from src.noising import alpha_delta
from src.utils import rng_for
from src.video_data import frame_difference_energy, make_moving_sequences
from src.video_model import (
    _ll_covariance,
    fit_ll_ar1,
    fit_video_tree,
    ll_log_likelihood,
    ll_posterior_mean,
)

_LOG_2PI = float(np.log(2.0 * np.pi))


def _smooth_frames(rng, n=40, side=32, sigma=2.0):
    """Spatially smooth stand-in frames.

    White noise will not do, and the reason is the point of the construction:
    the temporal coherence of a translating crop comes *entirely* from the
    spatial autocorrelation of the frame at the displacement scale. Translate
    white noise by one pixel and successive frames are independent
    (frame-difference energy 2.0); translate a natural image and they are nearly
    identical (0.22). Testing the video pipeline on white noise would therefore
    test it in the one regime where there is no temporal structure to find.
    """
    base = gaussian_filter(
        rng.standard_normal((n, side, side)), sigma=(0, sigma, sigma)
    )
    return base / base.std()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def test_moving_sequences_preserve_frame_statistics():
    """Translation of a reflect-padded canvas must not change the pixel marginals.

    If it did, every subband statistic measured on video would differ from the
    image case for reasons that have nothing to do with motion.
    """
    rng = rng_for("video-data")
    imgs = _smooth_frames(rng)
    vids = make_moving_sequences(imgs, 6, rng)
    assert vids.shape == (40, 6, 32, 32)
    assert abs(vids.std() - imgs.std()) < 0.05
    assert abs(float(vids.mean()) - float(imgs.mean())) < 0.05


def test_moving_sequences_are_temporally_coherent_and_actually_move():
    rng = rng_for("video-data-coherent")
    imgs = _smooth_frames(rng)
    vids = make_moving_sequences(imgs, 6, rng)
    # Far below the value for independent frames (2.0)...
    assert frame_difference_energy(vids) < 1.5
    # ...but not zero, i.e. every sequence genuinely moves.
    assert frame_difference_energy(vids) > 0.01


def test_frame_difference_energy_hits_its_reference_values():
    rng = rng_for("video-fde")
    frozen = np.repeat(rng.standard_normal((20, 1, 8, 8)), 5, axis=1)
    assert frame_difference_energy(frozen) < 1e-12
    indep = rng.standard_normal((400, 5, 8, 8))
    assert abs(frame_difference_energy(indep) - 2.0) < 0.05


# ----------------------------------------------------------------------------
# The LL band, which never touches the grid
# ----------------------------------------------------------------------------

def test_ll_posterior_and_likelihood_match_a_direct_solve():
    rng = rng_for("video-ll")
    f_len, mean, var, rho = 6, 1.5, 4.0, 0.9
    sigma = _ll_covariance(f_len, var, rho)
    a = mean + rng.standard_normal((50, f_len)) @ np.linalg.cholesky(sigma).T
    alpha, delta = alpha_delta(0.4)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    got = ll_posterior_mean(x, mean, var, rho, alpha, delta)
    obs = alpha**2 * sigma + delta * np.eye(f_len)
    want = mean + alpha * (sigma @ np.linalg.solve(obs, (x - alpha * mean).T)).T
    assert np.max(np.abs(got - want)) < 1e-12

    sign, logdet = np.linalg.slogdet(obs)
    c = x - alpha * mean
    quad = np.einsum("bi,ib->b", c, np.linalg.solve(obs, c.T))
    want_ll = float(np.sum(-0.5 * (quad + logdet + f_len * _LOG_2PI)))
    assert abs(ll_log_likelihood(x, mean, var, rho, alpha, delta) - want_ll) < 1e-8


def test_fit_ll_ar1_recovers_its_parameters():
    rng = rng_for("video-ll-fit")
    f_len, mean, var, rho = 8, -0.5, 2.0, 0.85
    sigma = _ll_covariance(f_len, var, rho)
    a = mean + rng.standard_normal((4000, f_len)) @ np.linalg.cholesky(sigma).T
    m, v, r = fit_ll_ar1(a)
    assert abs(m - mean) < 0.05
    assert abs(v - var) < 0.1
    assert abs(r - rho) < 0.03


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_recovers_temporal_correlation_and_ascends():
    """The whole pipeline on video with a known temporal structure.

    Sequences built by translating a real-statistics frame are strongly
    temporally correlated, so a fitted rho_time near 1 is the right answer and a
    rho_time near 0 would mean the temporal half is not wired up.
    """
    rng = rng_for("video-fit")
    imgs = _smooth_frames(rng, n=120)
    vids = make_moving_sequences(imgs, 5, rng)

    model, trace = fit_video_tree(
        vids, levels=5, t_train=[0.6],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.2, q=0.8),
        time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.3, q=0.8),
        n_iters=4, grid_size=121, chunk=16,
    )
    assert trace.monotone_violation <= 1e-6 * abs(trace.log_evidence[-1])
    assert model.k_time.rho > 0.5, f"rho_time {model.k_time.rho}"
    assert np.isfinite(model.log_likelihood_videos(vids[:20], 0.6))


@pytest.mark.slow
def test_freezing_time_holds_rho_at_zero_and_costs_coherence():
    """The control must actually be a control.

    `freeze_time` has to leave rho_time exactly where it started, and the
    resulting samples must be *less* temporally coherent than the fitted model's.
    Without this, the comparison in exp_26 could be reporting a difference that
    is not the one it claims.
    """
    rng = rng_for("video-frozen")
    imgs = _smooth_frames(rng, n=100)
    vids = make_moving_sequences(imgs, 5, rng)
    common = dict(
        levels=5, t_train=[0.6],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.2, q=0.8),
        n_iters=3, grid_size=121, chunk=16,
    )
    fitted, _ = fit_video_tree(
        vids, time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.3, q=0.8),
        **common,
    )
    frozen, _ = fit_video_tree(
        vids, time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.0, q=0.8),
        freeze_time=True, **common,
    )
    assert frozen.k_time.rho == 0.0

    a = frame_difference_energy(fitted.sample_ancestral(200, 5, rng))
    b = frame_difference_energy(frozen.sample_ancestral(200, 5, rng))
    assert a < b, f"fitted {a:.4f} should be more coherent than frozen {b:.4f}"


@pytest.mark.slow
def test_cut_model_fits_and_beats_the_caterpillar_on_coherence():
    """The space-for-time trade, end to end.

    Cutting the spatial tree below depth 1 severs 12 edges per frame and buys a
    temporal edge for each of the four subtrees per orientation. The prediction
    (§9.2 of the write-up) is that the coupled variance fraction rises from
    ~0.47 to ~0.65, so generated video must come out *more* temporally coherent
    -- a lower frame-difference energy -- than the plain caterpillar's.
    """
    rng = rng_for("video-cut-fit")
    imgs = _smooth_frames(rng, n=120)
    vids = make_moving_sequences(imgs, 5, rng)
    common = dict(
        levels=5, t_train=[0.6],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.2, q=0.8),
        time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.3, q=0.8),
        n_iters=4, grid_size=121, chunk=16,
    )
    flat, _ = fit_video_tree(vids, cut_depth=0, **common)
    cut, _ = fit_video_tree(vids, cut_depth=1, **common)

    assert cut.cut_depth == 1 and cut.k_time_sub is not None
    a = frame_difference_energy(flat.sample_ancestral(300, 5, rng))
    b = frame_difference_energy(cut.sample_ancestral(300, 5, rng))
    assert b < a, f"cut model {b:.4f} should be more coherent than caterpillar {a:.4f}"


@pytest.mark.slow
def test_cut_model_em_is_monotone():
    rng = rng_for("video-cut-monotone")
    imgs = _smooth_frames(rng, n=100)
    vids = make_moving_sequences(imgs, 5, rng)
    _, trace = fit_video_tree(
        vids, levels=5, t_train=[0.6],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.2, q=0.8),
        time_kernel_factory=lambda r: GaussianAR1Kernel(rho=0.3, q=0.8),
        n_iters=5, grid_size=121, chunk=16, cut_depth=1,
    )
    assert trace.monotone_violation <= 1e-6 * abs(trace.log_evidence[-1])
