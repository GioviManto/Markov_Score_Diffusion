"""Fitting the wavelet HMT: the Gaussian control, end to end.

The point of these tests is the control the project already uses everywhere
else: on data generated from a *Gaussian* tree, the learned per-scale kernel must
recover the generating parameters, and EM must ascend monotonically. If that
fails, nothing measured on real images means anything.

The data is built in wavelet space and pushed through the inverse transform, so
the test exercises the full pipeline -- synthesis, analysis, standardisation,
per-orientation BP, per-level M-step -- rather than the BP alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hierarchy import TreeIndex
from src.kernels import GaussianAR1Kernel
from src.utils import rng_for
from src.wavelet import WaveletQuadtree, tree_to_images
from src.wavelet_model import SubbandScales, fit_wavelet_tree

LEVELS = 4          # 16x16 images: depth-3 quadtrees, 85 nodes per orientation
TRUE_RHO = [0.75, 0.6, 0.45]


def _sample_tree_nodes(rng, ti: TreeIndex, n: int, rhos) -> np.ndarray:
    """(n, n_nodes) from the unit-variance tree recursion with per-level rho."""
    out = np.empty((n, ti.n_nodes))
    out[:, 0] = rng.standard_normal(n)
    for d in range(ti.depth):
        rho = rhos[d]
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        parent_vals = np.repeat(out[:, parents], ti.branching, axis=1)
        out[:, kids] = (
            rho * parent_vals
            + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, len(kids)))
        )
    return out


def _make_images(rng, n: int):
    qt = WaveletQuadtree(side=2**LEVELS, levels=LEVELS)
    ti = TreeIndex(qt.depth, 4)
    nodes = np.stack(
        [_sample_tree_nodes(rng, ti, n, TRUE_RHO) for _ in range(3)], axis=1
    )
    scaling = rng.standard_normal((n, 1))
    return qt, tree_to_images(qt, nodes, scaling), nodes


def test_subband_scales_round_trip():
    rng = rng_for("wavelet-model-scales")
    qt, _, nodes = _make_images(rng, 64)
    sc = SubbandScales.fit(qt, nodes)
    assert sc.scales.shape == (3, qt.depth + 1)
    back = sc.restore(qt, sc.standardise(qt, nodes))
    assert np.max(np.abs(back - nodes)) < 1e-12


def test_delta_by_depth_scales_as_one_over_variance():
    rng = rng_for("wavelet-model-delta")
    qt, _, nodes = _make_images(rng, 64)
    sc = SubbandScales.fit(qt, nodes)
    got = sc.delta_by_depth(0, 0.5)
    assert np.allclose(got, 0.5 / sc.scales[0] ** 2)


def test_grid_sizing_spends_points_where_the_likelihood_is_narrow():
    """The sizing rule, on scales shaped like a real image's.

    Coarse subbands have the largest scales and therefore the narrowest
    likelihoods, so they must get the *most* points despite holding the fewest
    nodes. That inversion is the whole idea.
    """
    from src.wavelet_model import per_depth_grid_sizes

    scales = np.array([[10.7, 5.1, 2.3, 1.0, 0.38]] * 3)
    sizes = per_depth_grid_sizes(scales, t_min=0.05)
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[0] > 10 * sizes[-1]
    assert all(s % 2 == 1 for s in sizes), "0 must lie on every grid"

    # Relaxing the target t asks for less resolution everywhere.
    looser = per_depth_grid_sizes(scales, t_min=0.3)
    assert all(a <= b for a, b in zip(looser, sizes))

    # The state constraint floors the fine end rather than letting it collapse.
    assert min(per_depth_grid_sizes(scales, t_min=2.0)) >= 33


def _ladder_images(rng, n, ladder=(10.0, 5.0, 2.3, 1.0)):
    """Images whose subband scales span decades, as a real image's do.

    `_make_images` gives every subband unit variance, and on *that* data a
    uniform grid is perfectly adequate -- which is exactly why the per-depth
    machinery cannot be tested on it. The resolution problem is created by the
    spread of the scales, so the fixture has to have one.
    """
    qt = WaveletQuadtree(side=2**LEVELS, levels=LEVELS)
    ti = TreeIndex(qt.depth, 4)
    nodes = np.stack(
        [_sample_tree_nodes(rng, ti, n, TRUE_RHO) for _ in range(3)], axis=1
    )
    depth_of = qt.node_depth
    for d, scale in enumerate(ladder):
        nodes[:, :, depth_of == d] *= scale
    scaling = rng.standard_normal((n, 1))
    return qt, tree_to_images(qt, nodes, scaling)


def test_per_depth_grids_resolve_small_t_where_a_uniform_grid_cannot():
    """The fix, stated as the thing it was supposed to buy.

    A uniform grid of comparable total size cannot resolve the coarse subbands
    below t ~ 0.9; a per-depth grid sized for t = 0.05 does, and the same
    `resolution_report` says so.
    """
    rng = rng_for("wavelet-model-resolve")
    qt, images = _ladder_images(rng, 80)

    per_depth, _ = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.5],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.6),
        n_iters=1, t_resolve=0.05, chunk=64,
    )
    uniform, _ = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.5],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.6),
        n_iters=1, grid_size=241, chunk=64,
    )
    assert per_depth.resolution_report(0.05)["min_resolved_t"] <= 0.05
    assert per_depth.resolution_report(0.05)["resolved"]
    assert uniform.resolution_report(0.05)["min_resolved_t"] > 0.2
    assert not uniform.resolution_report(0.05)["resolved"]


def test_grid_size_and_t_resolve_are_mutually_exclusive():
    rng = rng_for("wavelet-model-exclusive")
    _, images, _ = _make_images(rng, 8)
    with pytest.raises(ValueError, match="not both"):
        fit_wavelet_tree(
            images, levels=LEVELS, t_train=[0.5],
            kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.6),
            n_iters=1, grid_size=101, t_resolve=0.1,
        )


@pytest.mark.slow
def test_gaussian_control_recovers_per_level_rho_and_ascends():
    """The negative control: Gaussian tree in, Gaussian tree out.

    Each level's rho is recovered from noisy observations at three noise levels
    at once, which is the regime the model is meant to work in -- one kernel
    serving every t.
    """
    rng = rng_for("wavelet-model-control")
    qt, images, _ = _make_images(rng, 400)

    model, trace = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.2, 0.5, 1.0],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.5),
        n_iters=12, half_width=8.0, grid_size=241, chunk=64,
    )

    assert trace.monotone_violation <= 1e-6 * abs(trace.log_evidence[-1])
    got = [model.kernels[0][d].rho for d in range(qt.depth)]
    assert np.max(np.abs(np.array(got) - np.array(TRUE_RHO))) < 0.05

    # And the innovation variance must land near 1 - rho^2, the value that keeps
    # every level at unit marginal variance. The tolerance here is deliberately
    # looser than the one on rho: the project's own measurement (advisor document
    # question D) is that Fisher information per sequence for the innovation
    # *variance* falls 142x between t = 0.05 and t = 1.6, against 26x for the
    # correlation. Higher-order structure is far more fragile under the channel,
    # so at this sample size q is expected to be the noisier of the two.
    q = np.array([model.kernels[0][d].q for d in range(qt.depth)])
    assert np.max(np.abs(q - (1.0 - np.array(TRUE_RHO) ** 2))) < 0.08


@pytest.mark.slow
def test_reverse_samples_agree_with_ancestral_samples():
    """Reverse diffusion and ancestral sampling target the same distribution.

    This is the regression test for a real bug: `sample_reverse` used to return
    `x(t_min)` rather than the posterior-mean readout. Because `t_min` is floored
    at the grid-resolved t -- around 0.9 for natural-image subband scales, where
    Delta_t is still ~0.8 -- the returned samples carried most of a unit of noise,
    and comparing them against clean ancestral samples showed standard-deviation
    gaps of 2 and worse. The gap was entirely the un-removed noise.
    """
    rng = rng_for("wavelet-model-reverse")
    qt, images, _ = _make_images(rng, 300)
    model, _ = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.4],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.6),
        n_iters=10, half_width=8.0, grid_size=241, chunk=64,
    )

    anc = model.sample_ancestral(600, rng)
    rev = model.sample_reverse(48, rng, n_steps=60, t_max=3.0, chunk=48)

    assert np.all(np.isfinite(rev))
    # Pixel-level spread must match to well within the ~10% sampling error of
    # 48 samples; before the fix this ratio was off by a factor of order 2.
    ratio = float(rev.std() / anc.std())
    assert 0.8 < ratio < 1.25, f"reverse/ancestral pixel std ratio {ratio:.3f}"


@pytest.mark.slow
def test_denoising_beats_the_observation_and_likelihood_is_finite():
    rng = rng_for("wavelet-model-denoise")
    qt, images, _ = _make_images(rng, 200)
    model, _ = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.5],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.5),
        n_iters=8, half_width=8.0, grid_size=241, chunk=64,
    )
    from src.noising import alpha_delta

    alpha, delta = alpha_delta(0.5)
    test = _make_images(rng_for("wavelet-model-denoise-test"), 64)[1]
    noisy = alpha * test + np.sqrt(delta) * rng.standard_normal(test.shape)

    hat = model.denoise_images(noisy, 0.5)
    err_model = float(np.mean((hat - test) ** 2))
    err_raw = float(np.mean((noisy / alpha - test) ** 2))
    assert err_model < err_raw

    ll = model.log_likelihood_images(noisy, 0.5)
    assert np.isfinite(ll)


@pytest.mark.slow
def test_per_image_likelihood_sums_to_the_total():
    """The vector must reconstruct the scalar exactly, not approximately.

    A per-image likelihood that only *roughly* sums to the reported total would
    be an apportionment rather than a decomposition, and every standard error
    computed from it would be describing the apportionment. Both corrections that
    turn tree evidence into an image likelihood -- the standardisation Jacobian
    and the scaling-coefficient term -- are genuinely per-image, so exactness is
    achievable and is the right thing to require.
    """
    rng = rng_for("wavelet-model-perimage")
    qt, images, _ = _make_images(rng, 120)
    model, _ = fit_wavelet_tree(
        images, levels=LEVELS, t_train=[0.5],
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.5),
        n_iters=4, half_width=8.0, grid_size=161, chunk=64,
    )
    test = _make_images(rng_for("wavelet-model-perimage-test"), 32)[1]

    total = model.log_likelihood_images(test, 0.5)
    each = model.log_likelihood_images(test, 0.5, per_image=True)

    assert each.shape == (len(test),)
    assert np.all(np.isfinite(each))
    assert float(np.sum(each)) == pytest.approx(total, rel=1e-12)


@pytest.mark.slow
def test_per_image_likelihood_gives_a_paired_comparison_an_error_bar():
    """What the vector is for: two models on the SAME images are paired.

    The paired standard error is far smaller than either model's spread across
    images, because image difficulty is common to both and cancels. That is the
    entire reason a likelihood can resolve a model difference that an unpaired
    sample statistic cannot, so it is asserted rather than assumed.
    """
    rng = rng_for("wavelet-model-paired")
    qt, images, _ = _make_images(rng, 120)
    common = dict(images=images, levels=LEVELS, t_train=[0.5], n_iters=4,
                  half_width=8.0, grid_size=161, chunk=64)
    weak, _ = fit_wavelet_tree(
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.0, q=1.0), **common
    )
    fitted, _ = fit_wavelet_tree(
        kernel_factory=lambda d, r: GaussianAR1Kernel(rho=0.3, q=0.5), **common
    )
    test = _make_images(rng_for("wavelet-model-paired-test"), 48)[1]

    a = fitted.log_likelihood_images(test, 0.5, per_image=True)
    b = weak.log_likelihood_images(test, 0.5, per_image=True)
    diff = a - b

    paired_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    unpaired_se = float(
        np.sqrt(np.var(a, ddof=1) + np.var(b, ddof=1)) / np.sqrt(len(diff))
    )
    assert paired_se < unpaired_se, (
        f"pairing bought nothing: {paired_se:.4g} vs {unpaired_se:.4g}"
    )
