"""Video sequences with real frame statistics and controlled motion.

No video dataset is downloaded here, and the honest description of what this
produces is **real natural-image statistics with synthetic rigid motion**. That
is a genuine limitation and it is stated in the results rather than implied
away: real video has non-rigid motion, occlusion, and lighting change, none of
which this has.

What it does have is the property the experiment actually needs -- temporal
dependence that is real, strong, and of known origin -- so that "does modelling
time help?" has an answer that can be checked rather than believed. A model that
cannot exploit motion this clean will not exploit real motion either.

Construction: reflect-pad each CIFAR image, then crop a 32x32 window that
translates at constant integer velocity. Integer translation of a padded canvas
keeps the pixel marginals of a real photograph exactly, and moving the *window*
(rather than rolling the image) means content genuinely enters and leaves, so the
coarse coefficients vary over time instead of being trivially constant -- which
is what a wraparound roll would give, and would make the temporal problem
degenerate.

**Where the temporal coherence comes from.** Entirely from the *spatial*
autocorrelation of the frame at the displacement scale, and nothing else. Shift
a natural image by one pixel and successive frames are nearly identical
(frame-difference energy ~0.22); shift white noise by one pixel and they are
independent (~2.0, the same as unrelated frames). So the motion supplies no
temporal structure by itself -- it converts spatial smoothness into temporal
smoothness. Two consequences worth keeping in view: the coherence here is a
property of CIFAR rather than of the construction, and a white-noise control
would exercise this code in the one regime where there is nothing to learn.
"""

from __future__ import annotations

import numpy as np


def make_moving_sequences(
    images: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    max_speed: int = 1,
) -> np.ndarray:
    """(B, N, N) images -> (B, F, N, N) translating crops.

    Each sequence gets one constant velocity drawn from
    {-max_speed..max_speed}^2, excluding (0, 0) so every sequence actually moves.
    """
    images = np.asarray(images, dtype=float)
    b, n, _ = images.shape
    pad = max_speed * (n_frames - 1) + 1
    canvas = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

    speeds = np.arange(-max_speed, max_speed + 1)
    vel = rng.choice(speeds, size=(b, 2))
    still = (vel == 0).all(axis=1)
    while still.any():
        vel[still] = rng.choice(speeds, size=(still.sum(), 2))
        still = (vel == 0).all(axis=1)

    out = np.empty((b, n_frames, n, n))
    for i in range(b):
        vy, vx = int(vel[i, 0]), int(vel[i, 1])
        for f in range(n_frames):
            r0 = pad + vy * f
            c0 = pad + vx * f
            out[i, f] = canvas[i, r0 : r0 + n, c0 : c0 + n]
    return out


def frame_difference_energy(videos: np.ndarray) -> float:
    """Mean squared successive-frame difference, normalised by frame variance.

    The single most direct measure of temporal coherence, and the one the
    generated video has to reproduce: 0 is a frozen sequence, 2 is successive
    frames independent (Var(a - b) = 2 Var for independent equal-variance
    frames). Anything a model generates without temporal coupling lands at 2.
    """
    v = np.asarray(videos, dtype=float)
    diff = np.diff(v, axis=1)
    return float(np.mean(diff**2) / np.mean(v**2))
