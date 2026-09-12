"""The rotating-ring model: a chain whose dynamics is invisible to its marginals.

Where this comes from
---------------------
This is Problem 1 of `research/board-3problems`, ported here so the paper's
package is self-contained. The closed forms below are audited in that
directory against independent quadrature (`code/audit_three_problems.py`,
check P1d, ψ-posterior against quadrature at 1e-9; check P1a-c, the closed-form
score against a polar-quadrature reference). `tests/test_ring.py` re-checks the
two properties this paper actually leans on.

The model
---------
A trajectory is T frames in the plane, so the ambient dimension is D = 2T:

    r_0 ~ rho * exp(-(rho-1)^2 / 2*lam)      (a ring of radius ~1, width ~sqrt(lam))
    theta_0 ~ Uniform[0, 2*pi)
    z_0 = r_0 * (cos theta_0, sin theta_0)
    z_{u+1} = R_psi z_u + sigma * eta_u ,    eta_u ~ N(0, I_2)

`R_psi` is rotation by psi. One sample is a WHOLE trajectory, and the diffusion
channel hits all of it at once:

    x = m a + sqrt(Delta) xi ,   m = e^{-t},  Delta = 1 - e^{-2t}

Two facts make this the right fourth rung
-----------------------------------------
1. **The rotation is a gauge.** `U_psi = blockdiag(R_{-u psi})` is orthogonal,
   so it commutes with the isotropic channel and maps the model onto psi = 0 --
   a plain 2-D random walk started on the ring:

       P_t(x | psi) = P_t^0(U_psi x)

   So the psi-conditional density needs no new integral: gauge the data and
   evaluate the psi = 0 density. That is `log_pt_conditional` below.

2. **Marginal blindness (a theorem, not an approximation).** Since
   `z_u = R^u z_0 + sigma * sum_{k<=u} R^{u-k} eta_k`, with each
   `R^{u-k} eta_k =d eta_k` by isotropy and `R^u z_0 =d z_0` by the uniform
   phase,

       z_u  =d  z_0 + sigma * sqrt(u) * xi        -- psi does not appear

   Every single-frame marginal is psi-free at every noise level. A per-frame
   model therefore carries EXACTLY ZERO information about the dynamics -- not
   less, zero. `log_pt_marginal` below computes that psi-free marginal, and the
   control arm of `exp_28_ring_em` uses it to show that EM on marginals cannot
   move p(psi) off its initialisation while EM on the joint recovers it.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ive

__all__ = [
    "RingConfig",
    "rot",
    "gauge",
    "sample_trajectories",
    "noise",
    "log_pt_conditional",
    "log_pt_marginal",
]


class RingConfig:
    """Ring geometry, walk noise, and the radial quadrature they induce.

    The ring prior is the quadratic confining potential written on the board,
    `V(r) = (r - r_star)^2 / 2 lam`, whose Boltzmann weight times the
    two-dimensional area element gives the radial density

        p(rho)  proportional to  rho * exp(-(rho - r_star)^2 / 2 lam) .

    `r_star` is the radius the well sits at and `lam` its width (temperature).
    Both are free, because Rung 4a estimates them.

    The radial integral is done once, with Gauss-Legendre nodes on [0, r_max];
    `r_max = r_star + 12*sqrt(lam)` puts the truncation twelve standard
    deviations beyond the well, past any mass.

    THE NORMALISER IS NOT OPTIONAL. `log_norm` below is
    `log int_0^inf rho exp(-(rho-r_star)^2 / 2 lam) drho`, which depends on
    BOTH `lam` and `r_star`. Any likelihood used to compare different
    `(lam, r_star)` must subtract it once per trajectory. Omitting it is not a
    small error: the density is unnormalised, so a wider well simply carries
    more mass and the profile likelihood increases without bound in `lam`,
    pinning the estimate at the top of whatever grid it is given. That failure
    is silent -- it produces a smooth, confident, wrong answer -- so the
    normaliser is applied unconditionally inside `log_pt_conditional` rather
    than left to a flag a caller can forget. It is a constant when `lam` and
    `r_star` are fixed, so it cancels in every psi-responsibility and costs
    nothing there. `tests/test_ring.py` pins both directions.
    """

    def __init__(self, lam: float = 0.05, sigma: float = 0.30,
                 r_star: float = 1.0, n_quad: int = 200):
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.r_star = float(r_star)
        x, w = leggauss(n_quad)
        r_max = self.r_star + 12.0 * np.sqrt(self.lam)
        self.r = 0.5 * r_max * (x + 1.0)
        self.w = 0.5 * r_max * w
        self.log_ring = -((self.r - self.r_star) ** 2) / (2.0 * self.lam)
        self.r_max = r_max

    @property
    def log_norm(self) -> float:
        """log of the ring density's normalising constant. See the class docstring."""
        return float(np.log(np.sum(self.w * self.r * np.exp(self.log_ring))))

    def __repr__(self) -> str:
        return (f"RingConfig(lam={self.lam:g}, sigma={self.sigma:g}, "
                f"r_star={self.r_star:g})")


def rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def K_min(n_frames: int) -> np.ndarray:
    """Covariance kernel of the driftless walk: K[u,v] = min(u, v).

    Row 0 vanishes -- frame 0 *is* z_0 and carries no walk noise -- so frame 0
    decouples exactly at every noise level.
    """
    u = np.arange(n_frames)
    return np.minimum.outer(u, u).astype(float)


def gauge(X: np.ndarray, psi: float) -> np.ndarray:
    """Apply U_psi: rotate frame u backwards by u*psi.

    X has shape (batch, T, 2). Undoes the rotation, mapping a psi-trajectory
    onto a psi = 0 one.
    """
    return np.stack(
        [X[:, u, :] @ rot(-u * psi).T for u in range(X.shape[1])], axis=1
    )


def sample_ring(n: int, cfg: RingConfig, rng: np.random.Generator) -> np.ndarray:
    """`n` draws of z_0 from the ring, by rejection against a fixed bound.

    The bound `r_max` is fixed rather than sample-dependent, so the acceptance
    region does not depend on the draw and the sampler is unbiased.
    """
    out, got = [], 0
    while got < n:
        r = rng.normal(cfg.r_star, np.sqrt(cfg.lam), size=2 * n + 64)
        r = r[(r > 0) & (r < cfg.r_max)]
        r = r[rng.random(r.size) < r / cfg.r_max]
        out.append(r)
        got += r.size
    r = np.concatenate(out)[:n]
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


def sample_trajectories(
    n: int,
    n_frames: int,
    psi_values: np.ndarray,
    cfg: RingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n` clean trajectories, one rotation angle per trajectory.

    `psi_values` has length `n`: it is the per-trajectory rotation, drawn by the
    caller from whatever p(psi) is the truth. Returns (a, psi_values) with `a`
    of shape (n, T, 2).
    """
    psi_values = np.asarray(psi_values, dtype=float)
    if psi_values.shape != (n,):
        raise ValueError(f"psi_values must have shape ({n},), got {psi_values.shape}")

    z = sample_ring(n, cfg, rng)
    traj = [z.copy()]
    for _ in range(n_frames - 1):
        nz = np.empty_like(z)
        # Group by distinct angle so each rotation matrix is built once.
        for psi in np.unique(psi_values):
            k = psi_values == psi
            nz[k] = z[k] @ rot(psi).T
        z = nz + cfg.sigma * rng.normal(size=z.shape)
        traj.append(z.copy())
    return np.stack(traj, axis=1), psi_values


def noise(a: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    """The OU channel, applied to the whole trajectory at once."""
    m, delta = np.exp(-t), 1.0 - np.exp(-2.0 * t)
    return m * a + np.sqrt(delta) * rng.normal(size=a.shape)


def _bessel_grid(m: float, beta_max: float, cfg: RingConfig, n_beta: int = 400):
    """The kappa-independent part of the radial integral, computed once.

    `_log_bessel_table` splits into

        z          = m * beta * rho          depends on (m, beta, rho)
        ive(0, z)                            depends on z only   <- the expensive part
        e          = log_ring - m^2 kappa rho^2 / 2 + |z|        <- depends on kappa

    so `ive(0, z)` can be shared across every value of kappa on a common beta
    grid. `log_pt_marginal` needs twelve different kappa (one per frame, since
    the walk variance grows with the frame index) and previously rebuilt the
    whole table twelve times, which made it ten times slower than the joint
    likelihood and, unlike it, insensitive to the number of trajectories --
    the tell that the cost was in the quadrature, not the data.

    Returns (beta_grid, log_ive, z) ready for `_reduce_bessel`.
    """
    b = np.linspace(0.0, beta_max, n_beta)
    z = m * b[:, None] * cfg.r[None, :]
    return b, ive(0, z), z


def _reduce_bessel(b, ive_z, z, kappa: float, m: float, cfg: RingConfig):
    """Finish `_bessel_grid` for one kappa. Cheap: no Bessel evaluation."""
    e = cfg.log_ring[None, :] - 0.5 * m**2 * kappa * cfg.r[None, :] ** 2 + np.abs(z)
    shift = e.max(axis=1)
    v = np.log(
        np.sum(cfg.w[None, :] * cfg.r[None, :] * ive_z * np.exp(e - shift[:, None]),
               axis=1)
    ) + shift
    return b, v


def _log_bessel_table(kappa: float, m: float, beta_max: float, cfg: RingConfig,
                      n_beta: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate log Z_t(beta) = log int rho e^{-(rho-1)^2/2lam} e^{-m^2 kappa rho^2/2} I_0(m rho beta) drho.

    `ive` is the exponentially scaled Bessel function, so the `exp(|z|)` factor
    is carried explicitly in the exponent and nothing overflows.
    """
    b = np.linspace(0.0, beta_max, n_beta)
    z = m * b[:, None] * cfg.r[None, :]
    e = (
        cfg.log_ring[None, :]
        - 0.5 * m**2 * kappa * cfg.r[None, :] ** 2
        + np.abs(z)
    )
    shift = e.max(axis=1)
    v = np.log(
        np.sum(
            cfg.w[None, :] * cfg.r[None, :] * ive(0, z) * np.exp(e - shift[:, None]),
            axis=1,
        )
    ) + shift
    return b, v


def log_pt_conditional(X: np.ndarray, t: float, psi: float,
                       cfg: RingConfig) -> np.ndarray:
    """log P_t(x | psi), normalised in every parameter this package estimates.

    Uses the gauge: rotate the data by -psi and evaluate the psi = 0 density.
    Conditionally on z_0 the trajectory is Gaussian with a z_0-independent
    covariance `A_t = m^2 sigma^2 K + Delta I`, so P_0 is a two-parameter
    *location* mixture of one Gaussian and the whole thing reduces to a
    quadratic form plus a single one-dimensional Bessel integral.

    X has shape (batch, T, 2); returns shape (batch,).

    Three terms, and which parameters each depends on:

      -0.5 * quad + logZ_t(beta)   psi, lam, r_star, sigma, t
      - log|A_t|                   sigma, t          (Gaussian normaliser, 2 dims)
      - cfg.log_norm               lam, r_star       (ring density normaliser)

    The last two are constants when only psi varies, so they cancel in every
    psi-responsibility and change nothing at Rungs 4b/4c. They are included
    anyway, because Rung 4a estimates `lam` and `r_star` and omitting
    `log_norm` there gives a likelihood that increases without bound in `lam` --
    a silent failure that returns a smooth, confident, wrong answer. Making the
    default correct is cheaper than remembering a flag.

    A global `log(2*pi)` per trajectory (the angular part of the z_0 density) is
    dropped: it depends on nothing and shifts every arm equally.
    """
    Xg = gauge(X, psi)
    n_frames = Xg.shape[1]
    m, delta = np.exp(-t), 1.0 - np.exp(-2.0 * t)

    A = m**2 * cfg.sigma**2 * K_min(n_frames) + delta * np.eye(n_frames)
    A_inv = np.linalg.inv(A)
    g = A_inv @ np.ones(n_frames)
    kappa = float(np.ones(n_frames) @ g)

    quad = np.einsum("bui,uv,bvi->b", Xg, A_inv, Xg)
    beta = np.linalg.norm(np.einsum("u,bui->bi", g, Xg), axis=1)

    bt, bv = _log_bessel_table(kappa, m, max(beta.max() * 1.02, 1e-9), cfg)
    # Two dimensions share A_t, hence log|A_t| rather than half of it.
    log_det = float(np.linalg.slogdet(A)[1])
    return -0.5 * quad + np.interp(beta, bt, bv) - log_det - cfg.log_norm


def log_pt_marginal(X: np.ndarray, t: float, cfg: RingConfig) -> np.ndarray:
    """Sum over frames of the per-frame marginal log-density -- the blind model.

    By marginal blindness, frame u has the law of `z_0 + sigma sqrt(u) xi`
    pushed through the channel, so its marginal is a ring of radius `m`
    convolved with an isotropic Gaussian of variance

        v_u = m^2 sigma^2 u + Delta

    and psi appears nowhere. This function therefore takes no `psi` argument:
    there is no psi to take. Any EM that scores trajectories with it is scoring
    them with a quantity that is constant in psi, so its responsibilities are
    exactly uniform.

    Returns shape (batch,), the sum over frames.
    """
    n_frames = X.shape[1]
    m, delta = np.exp(-t), 1.0 - np.exp(-2.0 * t)
    total = np.zeros(X.shape[0])

    # One shared Bessel grid for all frames. beta_u = |x_u| / v_u and v_u grows
    # with u, so the largest beta comes from whichever frame maximises that
    # ratio; take the max over frames and cover them all with one grid.
    radii = np.linalg.norm(X, axis=2)                       # (batch, T)
    v_all = m**2 * cfg.sigma**2 * np.arange(n_frames) + delta
    beta_max = float((radii / v_all[None, :]).max()) * 1.02
    b_grid, ive_z, z = _bessel_grid(m, max(beta_max, 1e-9), cfg)

    for u in range(n_frames):
        v_u = v_all[u]
        rad = radii[:, u]
        # p(x_u) ∝ ∫ rho e^{-(rho-r*)^2/2lam} exp(-(|x|^2 + m^2 rho^2)/2v) I_0(m rho |x| / v) drho
        # The |x|^2 term is pulled out of the integral; the rest is the same
        # Bessel table with kappa = 1/v and beta = |x|/v.
        beta = rad / v_u
        bt, bv = _reduce_bessel(b_grid, ive_z, z, 1.0 / v_u, m, cfg)
        # `- cfg.log_norm` ONCE PER FRAME, unlike `log_pt_conditional` which
        # subtracts it once. The asymmetry is not a slip: this model treats the
        # frames as independent, so it invokes the ring density n_frames times
        # and needs n_frames normalisers, whereas the joint model uses it once,
        # for z_0, and propagates. Getting this wrong sends the estimated well
        # width to infinity -- the blind arm reached lam = 3.5e6 before the
        # correction was added.
        total += (-0.5 * rad**2 / v_u + np.interp(beta, bt, bv)
                  - np.log(v_u) - cfg.log_norm)

    return total
