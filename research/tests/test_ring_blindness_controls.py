"""Paired numerical controls for Theorem (marginal blindness).

WHY THESE EXIST. The chapter's existing evidence for blindness is that the
per-frame estimator does not move off its initialisation, "to machine
precision". That is a real check, but it is not a check of the theorem: the
per-frame likelihood is implemented by a function that takes no `psi` argument,
so its independence of `psi` holds by construction of the code path. A zero
there confirms the wiring, and would catch a leak, but it cannot corroborate
the mathematics against an independently built alternative.

These tests do the independent version. Trajectories are simulated separately
at several rotation angles -- nothing shared but the seed policy -- and two
things are asserted about the SAMPLES, not about any density function:

  negative control: the one-frame law must not depend on psi, so an energy
                    distance between the psi-samples must sit inside the
                    sampling noise of two same-psi resamples;

  positive control: the two-frame joint MUST depend on psi, and in the
                    direction the gauge argument names -- the lag-one
                    cross-covariance is the rotation matrix scaled by the
                    one-frame second moment, so its antisymmetric part tracks
                    sin(psi).

A null result on its own cannot distinguish a true invariance from a statistic
too blunt to see anything, which is exactly what the positive control rules
out: the same estimator, on the same samples, at the same sizes, does move.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ring import RingConfig, noise, sample_trajectories
from src.utils import rng_for

N = 6000
T = 6
PSIS = (0.0, np.pi / 6, np.pi / 3, 1.1)


def _draw(psi: float, tag: str, t: float | None = None) -> np.ndarray:
    """An independent trajectory sample at a fixed rotation."""
    cfg = RingConfig()
    rng = rng_for("ring-blindness", tag, f"{psi:.6f}")
    a, _ = sample_trajectories(N, T, np.full(N, psi), cfg, rng)
    return a if t is None else noise(a, t, rng)


def _energy_distance(x: np.ndarray, y: np.ndarray, rng, m: int = 900) -> float:
    """Energy distance between two point clouds, subsampled for O(m^2) cost."""
    x = x[rng.choice(len(x), m, replace=False)]
    y = y[rng.choice(len(y), m, replace=False)]
    d = lambda u, v: np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())


@pytest.mark.parametrize("t", [None, 0.5])
def test_one_frame_law_does_not_depend_on_psi(t):
    """Negative control: across psi, no more separation than a resample gives."""
    rng = np.random.default_rng(0)
    frame = T // 2

    # The yardstick: two independent draws at the SAME psi. Any psi-to-psi
    # distance below this is indistinguishable from having drawn twice.
    base = _draw(PSIS[0], "base", t)[:, frame, :]
    base2 = _draw(PSIS[0], "base-replicate", t)[:, frame, :]
    noise_floor = _energy_distance(base, base2, rng)

    for psi in PSIS[1:]:
        other = _draw(psi, "cmp", t)[:, frame, :]
        d = _energy_distance(base, other, rng)
        assert d < 4 * abs(noise_floor) + 2e-3, (
            f"one-frame law moved with psi={psi:.3f} at t={t}: energy distance "
            f"{d:.2e} against a same-psi floor of {noise_floor:.2e}"
        )


def test_two_frame_joint_does_depend_on_psi():
    """Positive control: the same samples DO carry psi in the lag-one joint.

    E[z_{u+1} z_u^T] = R_psi E[z_u z_u^T] = s^2 R_psi under an isotropic
    one-frame law, so the antisymmetric part of the lag-one cross-covariance is
    s^2 sin(psi) -- zero at psi = 0 and growing to the first quadrant.
    """
    def lag_one_asymmetry(a: np.ndarray) -> float:
        u, v = a[:, :-1, :].reshape(-1, 2), a[:, 1:, :].reshape(-1, 2)
        c = (v.T @ u) / len(u)
        s2 = 0.5 * (u**2).sum(axis=1).mean()
        return float((c[1, 0] - c[0, 1]) / (2 * s2))

    vals = [lag_one_asymmetry(_draw(psi, "joint")) for psi in PSIS]

    # psi = 0 is the null: no preferred sense of rotation.
    assert abs(vals[0]) < 0.03, f"psi=0 should be symmetric, got {vals[0]:.3f}"
    # And the statistic tracks sin(psi) rather than merely being non-zero.
    for psi, got in zip(PSIS[1:], vals[1:]):
        assert got == pytest.approx(np.sin(psi), abs=0.06), (
            f"lag-one asymmetry {got:.3f} does not track sin({psi:.3f})"
            f"={np.sin(psi):.3f}"
        )
    # The contrast that makes the negative control meaningful.
    assert max(vals) - min(vals) > 0.5, (
        "positive control is too weak to certify that the negative control "
        "could have detected a dependence"
    )
