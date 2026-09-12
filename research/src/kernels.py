"""Parameterized transition kernels K_theta(a' | a) and their M-steps.

Each kernel exposes the same three things, which is all `em` ever asks for:

    log_matrix(grid)       -> (M, M)      log K[out, in] at the current theta
    grad_log_matrix(grid)  -> (P, M, M)   d log K[out, in] / d theta_p
    m_step(stats, grid)    -> kernel      the maximizer (or an ascent step) of
                                          Q(. | theta') = <Xi, log K_theta>

`log_matrix` is exactly the interface `priors.ChainPrior` already exposes, so a
learned kernel drops straight into `bp_grid.grid_bp_batch` and becomes a
denoiser at *every* noise level without any further fitting. That decoupling --
the parameters live on the clean chain, the noise level lives in the likelihood
-- is the structural reason this is cheaper than learning a score network.

Three rungs of expressivity, matching the three phases of the plan:

  GaussianAR1Kernel     rho, q             well specified only for a Gaussian
                                           chain; closed-form M-step; the
                                           correctness harness.
  LaplaceAR1Kernel      rho, b             well specified for the Laplace chain;
                                           closed-form M-step (weighted median +
                                           weighted mean absolute residual).
  MixtureInnovationKernel  rho, {w,m,s}_c  linear autoregression with an
                                           arbitrary innovation density; closed
                                           form M-step by an inner EM over the
                                           mixture label. This is the honest
                                           "we know it is Markov but not the
                                           model" estimator.
  MDNKernel             MLP weights        a small network emitting a
                                           state-dependent mixture; fully
                                           nonparametric in both the location
                                           and the shape of the transition.
                                           Gradient M-step, and the network is
                                           only ever evaluated on the M grid
                                           points -- never on the data.

Every kernel is immutable; `m_step` returns a new instance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from src.em import ExpectedStatistics
from src.nnet import MLP

_LOG_2PI = float(np.log(2.0 * np.pi))
_VAR_FLOOR = 1e-6

# How far below zero a per-edge Q gain may fall before the mixture ECM inner
# loop treats it as a defect rather than as floating-point noise. Q per edge is
# O(1) and accumulated over an (M, M) matrix, so rounding lives around 1e-13;
# 1e-10 is three orders above that and still far below the ~1e-7 plateau the
# loop actually stops at, so it cannot fire on a genuine small step.
_Q_GAIN_TOL = 1e-10


class ParametricKernel(Protocol):
    """Interface consumed by `em` and by `bp_grid` alike."""

    @property
    def name(self) -> str: ...

    @property
    def theta(self) -> np.ndarray: ...

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray: ...

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray: ...

    def m_step(
        self, stats: ExpectedStatistics, grid: np.ndarray
    ) -> "ParametricKernel": ...


def _residual(grid: np.ndarray, rho: float, grid_out: np.ndarray | None = None) -> np.ndarray:
    """e[out, in] = u_out - rho u_in, the innovation implied by a grid pair.

    `grid_out` lets the outgoing (child) axis live on a *different* grid from the
    incoming (parent) one, which is what makes a transition matrix rectangular.
    That is the whole mechanism behind per-depth grids: a subband whose
    likelihood is narrow needs a fine mesh, its neighbour may not, and the two
    are joined by an (M_out x M_in) matrix rather than a square one. Defaults to
    `grid`, so every existing square call is unchanged.
    """
    out = grid if grid_out is None else grid_out
    return out[:, None] - rho * grid[None, :]


# ----------------------------------------------------------------------------
# Rung 1: Gaussian AR(1) -- the closed-form correctness harness
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianAR1Kernel:
    """K(a'|a) = N(a'; rho a, q). theta = (rho, q)."""

    rho: float
    q: float

    @property
    def name(self) -> str:
        return "gaussian_ar1"

    @property
    def is_covariance_stationary(self) -> bool:
        """Whether the stationary reading of this fit is available at all.

        An AR(1) chain is covariance-stationary exactly when |rho| < 1. Outside
        that the finite-chain law is still valid but has no stationary
        distribution to match variances against, so `FROZEN.innovation_variance`
        and every "variance-matched across innovation families" statement stop
        applying to it.
        """
        return abs(self.rho) < 1.0

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.rho, self.q])

    def log_transition_matrix(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        e = _residual(grid, self.rho, grid_out)
        return -0.5 * e**2 / self.q - 0.5 * (_LOG_2PI + np.log(self.q))

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _residual(grid, self.rho)
        d_rho = e * grid[None, :] / self.q
        d_q = 0.5 * e**2 / self.q**2 - 0.5 / self.q
        return np.stack([d_rho, d_q])

    def m_step(
        self, stats: ExpectedStatistics, grid: np.ndarray,
        grid_out: np.ndarray | None = None,
    ) -> "GaussianAR1Kernel":
        """Exact maximizer: weighted least squares through the origin.

        With s_ab = sum_{jk} Xi[k, j] u_j^a u_k^b, setting grad Q = 0 gives
        rho = s_10... in the two-index notation below,

            rho = S01 / S00,   q = (S11 - S01^2 / S00) / W,

        the pairwise-belief-weighted Yule-Walker equations. Reducing to the
        clean-data limit (Xi a histogram of observed transitions) recovers the
        ordinary AR(1) least-squares estimator, as it must.

        NOTHING CONSTRAINS |rho| < 1 HERE, deliberately. The update is the exact
        maximiser of Q, and clamping it would trade an exact M-step for a
        silently biased one -- the outer EM would still ascend, so the damage
        would not show up in any monotonicity check. On a finite chain
        |rho| >= 1 is still a perfectly valid joint distribution, so there is no
        numerical problem either.

        What breaks at |rho| >= 1 is the INTERPRETATION, and only that. The
        chain is no longer covariance-stationary, so the variance-matching
        convention q = 1 - rho^2 that makes every innovation family share
        Cov(a_i, a_j) = rho^|i-j| stops meaning anything, and any statement in
        the paper that appeals to stationarity does not cover such a fit.
        `is_covariance_stationary` exposes the test, and the warning below
        exists so that a fit which has left that regime cannot be aggregated
        into a stationary claim without something having said so.
        """
        xi = stats.xi
        out = grid if grid_out is None else grid_out
        w_tot = float(xi.sum())
        s00 = float(np.einsum("kj,j,j->", xi, grid, grid))
        s01 = float(np.einsum("kj,j,k->", xi, grid, out))
        s11 = float(np.einsum("kj,k,k->", xi, out, out))
        rho = s01 / s00
        q = max((s11 - rho * s01) / w_tot, _VAR_FLOOR)
        fitted = GaussianAR1Kernel(rho=float(rho), q=float(q))
        if not fitted.is_covariance_stationary:
            warnings.warn(
                f"Gaussian M-step returned rho={rho:.4f} with |rho| >= 1. The "
                "fit is a valid finite-chain distribution and the M-step is "
                "still the exact maximiser, but the chain is not "
                "covariance-stationary and q = 1 - rho^2 no longer matches "
                "variances. Do not pool this fit into a claim that assumes "
                "stationarity.",
                RuntimeWarning,
                stacklevel=2,
            )
        return fitted


# ----------------------------------------------------------------------------
# Rung 2: Laplace AR(1) -- closed form despite the non-smooth kernel
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LaplaceAR1Kernel:
    """K(a'|a) = Laplace(a' - rho a; 0, b). theta = (rho, b)."""

    rho: float
    b: float

    @property
    def name(self) -> str:
        return "laplace_ar1"

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.rho, self.b])

    def log_transition_matrix(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        e = _residual(grid, self.rho, grid_out)
        return -np.abs(e) / self.b - np.log(2.0 * self.b)

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """Provided for the gradient-ascent route, but see the caveat.

        d/db is smooth and reproduces a finite difference of the exact evidence
        to machine precision. d/drho is not: it carries sign(e), a discontinuity
        across the diagonal, and trapezoidal quadrature of a discontinuous
        integrand loses the spectral accuracy the rest of this package enjoys.
        Measured against a finite difference of the grid evidence, the rho
        component is off by ~13% at M = 201 and still ~0.1% at M = 1601, decaying
        erratically rather than at a clean rate.

        This is a discretization artifact, not a violation of Fisher's identity
        -- the smooth direction confirms the identity holds -- but it means
        gradient ascent on a non-smooth kernel pays a grid tax that the exact
        M-step in `m_step` does not: that route touches Xi only through a
        weighted median, which is insensitive to the same discontinuity.
        """
        e = _residual(grid, self.rho)
        d_rho = np.sign(e) * grid[None, :] / self.b
        d_b = np.abs(e) / self.b**2 - 1.0 / self.b
        return np.stack([d_rho, d_b])

    def m_step(
        self, stats: ExpectedStatistics, grid: np.ndarray,
        grid_out: np.ndarray | None = None,
    ) -> "LaplaceAR1Kernel":
        """Exact maximizer, obtained by profiling b out of Q.

        Q(rho, b) = -(1/b) sum Xi |u_k - rho u_j| - W log(2b). For fixed rho the
        optimum is b(rho) = (1/W) sum Xi |u_k - rho u_j|, and substituting back
        gives Q(rho) = -W log(2 b(rho)) - W. Maximizing over rho is therefore
        *minimizing the belief-weighted mean absolute residual*: a weighted
        least-absolute-deviations regression through the origin, whose exact
        solution is the weighted median of the ratios u_k / u_j with weights
        Xi[k, j] |u_j|. So the joint M-step is closed form even though the
        kernel is not differentiable -- no line search, no gradient.

        Being a breakpoint estimator, `rho` is *lattice-valued*: on a uniform
        grid through the origin every ratio is a rational a/b with
        |a|,|b| <= (M-1)/2, and low-denominator values pool the weight of all
        their aliases (4/5 collects 8/10, 12/15, ...), which makes them
        attractors. At a truth of 0.7913 the estimate is pinned at exactly 0.8
        across an eightfold grid refinement, holding a constant 0.0087 bias --
        this is a bias, not a discretization error, and refining does not help.
        See Remark `rem:quantization` in report/em_bp_learning.tex,
        `outputs/exp_06_em_parameter_recovery/laplace_quantization.csv`, and
        tests/test_laplace_rho_lattice.py. It is also why exp_06's Laplace arm
        recovers rho to 1e-16: the truth there is 0.8 = 4/5, the strongest
        attractor in its neighbourhood, so that figure measures the lattice
        rather than the estimator's accuracy. The mixture kernel's rho block is
        a continuous weighted least-squares solve and is not affected.
        """
        xi = stats.xi
        out_g = grid if grid_out is None else grid_out
        w_tot = float(xi.sum())
        ratios = np.divide(
            out_g[:, None], grid[None, :],
            out=np.zeros((len(out_g), len(grid))),
            where=np.abs(grid[None, :]) > 0,
        )
        wts = xi * np.abs(grid)[None, :]
        mask = wts > 0
        rho = _weighted_median(ratios[mask].ravel(), wts[mask].ravel())
        resid = np.abs(_residual(grid, rho, out_g))
        b = max(float(np.sum(xi * resid)) / w_tot, np.sqrt(_VAR_FLOOR))
        return LaplaceAR1Kernel(rho=float(rho), b=float(b))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Value at which the cumulative weight first reaches half the total."""
    if values.size == 0:
        raise ValueError("Weighted median of an empty set.")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


# ----------------------------------------------------------------------------
# Rung 3: linear autoregression, unknown innovation density
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MixtureInnovationKernel:
    """K(a'|a) = sum_c pi_c N(a' - rho a; mu_c, s_c^2).

    Keeps the linear autoregression but makes the innovation law essentially
    free: a C-component mixture approximates heavy tails, skew and bimodality
    alike. This is the estimator to use when we genuinely do not know the model
    behind the data -- it never sees a Laplace density, yet it has to reproduce
    one from noisy observations.

    The component means are left unconstrained. Imposing the natural-looking
    zero-mean condition sum_c pi_c mu_c = 0 by recentering after each update is
    not harmless: the projection does not maximize Q and it broke monotone
    ascent by ~1e-2 nats in testing. Unconstrained ECM is exactly monotone, and
    on zero-mean data the fitted innovation mean lands at the 1e-3 level by
    itself, so the condition is reported as a diagnostic rather than enforced.
    """

    rho: float
    pi: np.ndarray
    mu: np.ndarray
    s2: np.ndarray

    # Inner-loop record, set by `m_step` on the kernel it returns. Defaults keep
    # every existing construction site working, and none of these enter `theta`,
    # so parameter counts and gradients are unaffected.
    inner_sweeps: int = 0
    inner_converged: bool = False
    inner_q_gain: float = float("nan")
    """How the ECM inner loop ended. `inner_converged=False` on a production fit
    means the M-step was still moving when it ran out of sweeps, so the outer
    iteration count is not the only budget that was binding."""

    @property
    def name(self) -> str:
        return f"mixture_innovation_C{len(self.pi)}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([[self.rho], self.pi, self.mu, self.s2])

    def scale_diagnostics(self, grid: np.ndarray) -> dict:
        """Whether the fitted mixture is resolved by the grid it lives on.

        `_VAR_FLOOR` is 1e-6, i.e. a component standard deviation floor of 1e-3, while the
        default grid spacing is h = 0.04. Nothing stops a component from being forty times
        narrower than one grid cell, and a high-capacity fit has every incentive to exploit
        that: an under-resolved spike can raise the quadrature likelihood without
        corresponding to any feature of the innovation law.

        That matters for a specific claim. The capacity sweep's improvement from C = 12 to
        C = 16 is small, and it cannot be read as real until the fitted components are known
        to be wider than the mesh. `s_min_over_h` is the number to check: below roughly 2 the
        narrowest component is a lattice artefact rather than a density.

        Reports alongside it the effective number of components actually in use --
        ``exp(entropy(pi))``, which counts a mixture that has collapsed onto two live
        components as 2 regardless of its nominal C -- and the worst deviation of a
        transition column from unit quadrature mass, weighted by nothing, since that is the
        same normalisation residual the grid diagnostics track.
        """
        h = float(grid[1] - grid[0])
        s = np.sqrt(self.s2)
        p = np.clip(self.pi, 1e-300, None)
        p = p / p.sum()

        K = np.exp(self.log_transition_matrix(grid))
        w = np.full(len(grid), h)
        w[0] *= 0.5
        w[-1] *= 0.5
        col_mass = (K * w[:, None]).sum(axis=0)
        interior = np.abs(grid) <= 0.5 * float(np.max(np.abs(grid)))

        return {
            "n_components": int(len(self.pi)),
            "grid_spacing": h,
            "s_min": float(s.min()),
            "s_median": float(np.median(s)),
            "s_min_over_h": float(s.min() / h),
            "s_median_over_h": float(np.median(s) / h),
            "resolved": bool(s.min() >= 2.0 * h),
            "min_weight": float(self.pi.min()),
            "effective_n_components": float(np.exp(-(p * np.log(p)).sum())),
            "column_mass_residual_max": float(np.max(np.abs(col_mass - 1.0))),
            "column_mass_residual_interior": float(
                np.max(np.abs(col_mass[interior] - 1.0))
            ),
        }

    @property
    def innovation_moments(self) -> dict:
        """Mean, variance and excess kurtosis of the fitted innovation law.

        Excess kurtosis is the project's scalar non-Gaussianity knob (0 for a
        Gaussian, +3 for the Laplace chain), so it is the natural single number
        to check a recovered innovation density against.
        """
        m1 = float(self.pi @ self.mu)
        cen = self.mu - m1
        m2 = float(self.pi @ (self.s2 + cen**2))
        m4 = float(self.pi @ (3.0 * self.s2**2 + 6.0 * self.s2 * cen**2 + cen**4))
        return {
            "innovation_mean": m1,
            "innovation_var": m2,
            "innovation_excess_kurtosis": m4 / m2**2 - 3.0 if m2 > 0 else float("nan"),
        }

    @classmethod
    def init(
        cls, n_components: int, rho: float, var: float, rng: np.random.Generator
    ) -> "MixtureInnovationKernel":
        """Spread components over the plausible innovation range."""
        pi = np.full(n_components, 1.0 / n_components)
        mu = np.linspace(-1.0, 1.0, n_components) * np.sqrt(var)
        mu = mu - mu @ pi
        if n_components == 1:
            mu = np.zeros(1)
        s2 = np.full(n_components, var) * (0.5 + rng.random(n_components))
        return cls(rho=rho, pi=pi, mu=mu, s2=s2)

    def _component_logs(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        """(C, M_out, M_in) log of pi_c N(e; mu_c, s2_c)."""
        e = _residual(grid, self.rho, grid_out)
        d = e[None, :, :] - self.mu[:, None, None]
        s2 = self.s2[:, None, None]
        return (
            np.log(self.pi)[:, None, None]
            - 0.5 * (_LOG_2PI + np.log(s2))
            - 0.5 * d**2 / s2
        )

    def log_transition_matrix(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        from scipy.special import logsumexp

        return logsumexp(self._component_logs(grid, grid_out), axis=0)

    def responsibilities(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        """(C, M_out, M_in) posterior over the mixture label given a transition."""
        comp = self._component_logs(grid, grid_out)
        return np.exp(comp - _logsumexp_axis0(comp)[None, :, :])

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. rho only; the mixture block uses its exact M-step.

        d log K / d rho = +u_j * sum_c r_c (e - mu_c) / s_c^2, where r_c are the
        responsibilities: the usual "gradient of a log-mixture is the
        responsibility-weighted average of the component gradients".

        The sign follows from e = u_k - rho u_j, so de/drho = -u_j, and

            d/drho log N(e; mu_c, s2_c) = -(e - mu_c)/s2_c * de/drho
                                        = +u_j (e - mu_c)/s2_c.

        The two minus signs cancel. This branch previously returned the negative
        of this quantity; the C = 1, mu = 0 case must agree with
        `GaussianAR1Kernel.grad_log_transition_matrix`, which uses the positive
        convention, and it now does. See
        `test_mixture_rho_gradient_matches_finite_difference`.
        """
        e = _residual(grid, self.rho)
        r = self.responsibilities(grid)
        d = e[None, :, :] - self.mu[:, None, None]
        inner = (r * d / self.s2[:, None, None]).sum(axis=0)
        return (grid[None, :] * inner)[None, :, :]

    def m_step(
        self, stats: ExpectedStatistics, grid: np.ndarray,
        grid_out: np.ndarray | None = None, n_inner: int = 16,
        inner_tol: float = 1e-8,
    ) -> "MixtureInnovationKernel":
        """Generalised (ECM) block M-step, alternating the mixture block and rho.

        This was called an "exact block M-step" while running exactly four fixed
        sweeps, which it is not: each block is a closed-form *conditional*
        maximiser, so a sweep ascends Q without maximising it, and four of them
        stop wherever four happens to land. That is an expectation/conditional-
        maximisation step -- generalised EM. The outer EM stays monotone either
        way, since ascending Q is all monotonicity needs, but "the maximiser of Q"
        was the wrong description and the fixed count was the wrong default for
        runs that scientific claims rest on.

        Now it iterates until the per-edge gain in Q falls below `inner_tol` or
        `n_inner` sweeps are spent, and records which happened on the returned
        kernel (`inner_sweeps`, `inner_converged`, `inner_q_gain`) so an unfinished
        inner loop is visible instead of silent.

        The default of 4 was materially under-converging the innovation SHAPE,
        which is the quantity every non-Gaussian claim depends on. Measured at
        C = 8, n = 256, t = 0.22, M = 401, fixed at 40 outer iterations, varying
        only this budget (true excess kurtosis is 3.0):

            n_inner    L/edge          rho       excess kurtosis   wall
              2       -1.2579069     0.81838          0.64          12 s
              4       -1.2571693     0.81865          1.68          17 s
              8       -1.2570541     0.81977          2.06          27 s
             16       -1.2570495     0.82006          2.14          48 s
             64       -1.2570679     0.82005          2.16         184 s

        Two things to read off it. The shape moves by 28% between 4 and 16 sweeps
        while rho barely moves at all -- the same rate asymmetry the outer loop
        shows, appearing again one level down. And the outer objective is NOT
        monotone in this budget: 64 sweeps lands slightly below 16.

        But the effect is confined to SHORT outer runs, and the first version of
        this note over-generalised from the 40-iteration measurement above. At 60
        outer iterations the inner budget stops mattering, because the two loops
        partially substitute -- the outer loop gets there anyway, just slower:

            n     n_inner   s_min/h   excess kurtosis
            128      4        9.58         3.217
            128     16       10.01         3.343
            512      4        9.43         2.967
            512     16        9.80         2.932

        Negligible, and non-monotone. So the honest statement is: 4 sweeps
        under-converges the shape when the outer loop is cut short, and does not
        when it is allowed to run. In the E9 efficiency run, which selects its
        outer count on validation from a grid up to 400, the median selection is
        120 and 70% of cells sit at 60 or above -- but 30% do not, and those are in
        the regime where this matters.

        Hence 16 as the default: it removes the interaction rather than requiring
        anyone to reason about which regime a cell is in, at a quarter of the
        64-sweep cost. `n_inner=4` remains a legitimate fast mode.

        Those measurements also settle the resolution question the same audit
        raised: s_min/h is 9.4-10.0 at C=8, i.e. the narrowest fitted component is
        nearly ten grid cells wide, five times the threshold below which a
        component is a lattice artefact. The production fits are resolved with
        room to spare, and `scale_diagnostics` is now recorded per cell so this is
        a checked fact rather than a spot check.

        On the sweep those numbers came from, the per-edge Q gain plateaus near
        1e-7 and stops falling, so the cap rather than the tolerance ends the
        loop. That was written up as the general case -- "`inner_converged` is
        normally False at these settings" -- and the frozen outputs say
        otherwise. Across the 1,728 certified exp_07 cells the flag is True in
        53%, and the rate climbs with the sample size:

            n_seq      32    128    512   2048   8192
            % early    15     22     57     78     96

        The n_seq=8192 column is measured from the run later withdrawn from the
        paper's headline table for a different reason (round-two review item 8:
        three of its source files were uncommitted -- exp_07_em_vs_score_network.py,
        frozen_config.py, src/em.py; tools/make_tab_efficiency.py and
        outputs/README_exp12_scaled.md have the detail). That withdrawal is
        about whether the score-error numbers in that row are reconstructible,
        not about whether inner sweeps went negative, and it does not implicate
        this signed-gain guard, which is a property of the M-step's algebra and
        not of which commit ran it. Flagged anyway, because the honest answer
        to "is the 96% figure re-verified under a clean deploy" is not yet.

        447 of those cells stop after a SINGLE sweep. So at the sizes carrying
        the headline the tolerance is what stops the loop, not the cap, and the
        opposite claim should not be relied on.

        That makes the sign of the gain load-bearing rather than academic, which
        is why the test below is signed. Instrumenting a faithful replay of the
        production fit (frozen config, n = 512 and n = 2048, 60 outer iterations)
        recorded 1,764 inner gains and *no* negative one, including across the
        41-of-60 outer iterations that stopped early at n = 2048. Every early
        stop measured so far is a genuine small positive step, which is what the
        conditional-maximiser argument predicts; the guard exists because
        nothing enforced it.

        The complete-data problem has a *second* latent variable, the mixture
        label, so the M-step is itself an EM whose responsibilities are computed
        against the fixed Xi. Both blocks are closed form:

            pi_c   = <Xi r_c> / <Xi>
            mu_c   = <Xi r_c e> / <Xi r_c>
            s2_c   = <Xi r_c (e - mu_c)^2> / <Xi r_c>
            rho    = <Xi sum_c r_c u_j (u_k - mu_c)/s2_c>
                     / <Xi sum_c r_c u_j^2 / s2_c>       (weighted least squares)

        Since Xi is fixed, every inner sweep is O(C M^2) on a matrix that never
        grows with the dataset -- which is why raising the cap costs little.
        """
        xi = stats.xi
        out_g = grid if grid_out is None else grid_out
        w_tot = float(xi.sum())
        current = self

        def q_per_edge(k: "MixtureInnovationKernel") -> float:
            """<Xi, log K> per edge. The objective the sweeps ascend."""
            return float((xi * k.log_transition_matrix(grid, out_g)).sum()) / w_tot

        q_prev = q_per_edge(current)
        sweeps, gain, converged = 0, float("inf"), False
        for _ in range(n_inner):
            e = _residual(grid, current.rho, out_g)
            r = current.responsibilities(grid, out_g)  # (C, M_out, M_in)
            wr = r * xi[None, :, :]
            mass = wr.sum(axis=(1, 2))
            mass = np.maximum(mass, 1e-300)
            pi = mass / w_tot
            mu = (wr * e[None, :, :]).sum(axis=(1, 2)) / mass
            dev = e[None, :, :] - mu[:, None, None]
            s2 = np.maximum((wr * dev**2).sum(axis=(1, 2)) / mass, _VAR_FLOOR)
            current = replace(current, pi=pi, mu=mu, s2=s2)

            # rho block: weighted least squares with component-specific offsets.
            r = current.responsibilities(grid, out_g)
            wr = r * xi[None, :, :]
            inv_s2 = 1.0 / current.s2[:, None, None]
            num = float(
                (wr * inv_s2 * grid[None, None, :]
                 * (out_g[None, :, None] - current.mu[:, None, None])).sum()
            )
            den = float((wr * inv_s2 * (grid**2)[None, None, :]).sum())
            current = replace(current, rho=num / den)

            sweeps += 1
            q_now = q_per_edge(current)
            gain = q_now - q_prev
            q_prev = q_now
            # Signed, like the outer loop's stopping test. `gain <= inner_tol`
            # accepted a DECREASE as convergence and returned the descending
            # iterate -- the same defect `fit_em` was corrected for one level
            # up, missed one level down. Each block here is a closed-form
            # conditional maximiser, so Q ascends by construction and a
            # materially negative gain means the block algebra is wrong, not
            # that the loop has finished. Raise rather than roll back: a
            # broken M-step should stop a fit, not quietly return the last
            # good iterate and let the run look successful.
            if gain < -_Q_GAIN_TOL:
                raise FloatingPointError(
                    f"mixture ECM inner sweep {sweeps} decreased Q by "
                    f"{-gain:.3e} per edge (tolerance {_Q_GAIN_TOL:.1e}). "
                    "Each block is a conditional maximiser, so this is a "
                    "defect in the block updates, not a convergence signal."
                )
            if gain <= inner_tol:
                converged = True
                break

        return replace(
            current, inner_sweeps=sweeps, inner_converged=converged,
            inner_q_gain=float(gain),
        )


# ----------------------------------------------------------------------------
# Rung 4: neural transition kernel (mixture-density network)
# ----------------------------------------------------------------------------

@dataclass
class MDNKernel:
    """K(a'|a) = sum_c pi_c(a) N(a'; m_c(a), s_c(a)^2), heads from a small MLP.

    This is the "small deep network" rung, but placed *inside* the exact BP
    recursion rather than replacing it. The payoff is visible in the M-step: by
    Fisher's identity the gradient of the marginal log-likelihood factors as

        dL/dphi = sum_j [ sum_k Xi[k, j] d log K[k,j] / d head(u_j) ] d head(u_j)/dphi,

    so the network is evaluated and backpropagated *only at the M grid points*,
    once per EM iteration, no matter how large the dataset is. Data enters
    exclusively through Xi. A vanilla score network, by contrast, must
    backpropagate through every training example at every step.
    """

    net: MLP
    n_components: int
    lr: float = 5e-2
    backtrack_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    _adam: dict = None  # type: ignore[assignment]

    @classmethod
    def init(
        cls,
        n_components: int,
        hidden: int,
        rng: np.random.Generator,
        lr: float = 5e-2,
    ) -> "MDNKernel":
        net = MLP.init((1, hidden, hidden, 3 * n_components), rng)
        return cls(net=net, n_components=n_components, lr=lr)

    @property
    def name(self) -> str:
        return f"mdn_C{self.n_components}_P{self.net.n_params}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.net.params])

    def _heads(self, grid: np.ndarray):
        """Return (log_pi, m, log_s2) each (C, M) plus the forward cache."""
        out, cache = self.net.forward(grid[:, None])
        c = self.n_components
        logit, m, log_s2 = out[:, :c], out[:, c : 2 * c], out[:, 2 * c :]
        log_pi = logit - _logsumexp_rows(logit)[:, None]
        log_s2 = np.clip(log_s2, -12.0, 6.0)
        return log_pi.T, m.T, log_s2.T, cache

    def _component_logs(self, grid: np.ndarray) -> np.ndarray:
        """(C, M_out, M_in): log pi_c(u_in) N(u_out; m_c(u_in), s_c(u_in)^2)."""
        log_pi, m, log_s2, _ = self._heads(grid)
        d = grid[None, :, None] - m[:, None, :]
        return (
            log_pi[:, None, :]
            - 0.5 * (_LOG_2PI + log_s2[:, None, :])
            - 0.5 * d**2 * np.exp(-log_s2)[:, None, :]
        )

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """Grid-normalized log kernel: each column integrates to one under the
        same trapezoidal rule BP uses.

        Without the column normalization the quadrature mass of K would differ
        from one by the grid's truncation error, and that error would show up as
        a spurious drift in the reported evidence. Normalizing makes the fitted
        object a proper density *of the discretized model*, which is the model
        whose likelihood EM is actually maximizing.
        """
        log_k = _logsumexp_axis0(self._component_logs(grid))
        log_w = np.log(_quad_weights(grid))
        return log_k - _logsumexp_axis0(log_k + log_w[:, None])[None, :]

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MDNKernel uses the fused gradient inside m_step; the (P, M, M) "
            "tensor would be P times larger than the kernel itself."
        )

    def m_step(self, stats: ExpectedStatistics, grid: np.ndarray) -> "MDNKernel":
        """One backtracked Adam ascent step on Q -- a generalized M-step.

        Generalized EM requires Q to *increase*, not to be maximized. A fixed
        learning rate does not deliver that: run without the backtracking below,
        this kernel raised the marginal log-likelihood overall but violated
        monotonicity by hundreds of nats along the way, at which point EM's one
        guarantee is gone and a diverging run is indistinguishable from a slow
        one. We therefore evaluate Q after the step and halve it until the step
        is an ascent, which is cheap: Q is the contraction <Xi, log K>, and Xi
        is already in hand.

        The gradient uses the centered statistic

            Xi~[k, j] = Xi[k, j] - (sum_k Xi[k, j]) p_j(k),

        where p_j is the grid-normalized kernel column. The subtracted term is
        the derivative of the column log-partition, i.e. the familiar
        "observed minus expected" form of an exponential-family gradient; it is
        what makes the ascent respect the normalization imposed above. Note
        that Xi~ has exactly zero column sums, which is why the softmax head
        needs no further mean subtraction.
        """
        comp = self._component_logs(grid)  # (C, M_out, M_in), unnormalized in k
        resp = np.exp(comp - _logsumexp_axis0(comp)[None, :, :])
        _, m, log_s2, cache = self._heads(grid)

        log_k = _logsumexp_axis0(comp)
        log_w = np.log(_quad_weights(grid))
        p_col = np.exp(
            log_k + log_w[:, None] - _logsumexp_axis0(log_k + log_w[:, None])[None, :]
        )
        xi = stats.xi - stats.xi.sum(axis=0)[None, :] * p_col

        d = grid[None, :, None] - m[:, None, :]
        inv_s2 = np.exp(-log_s2)[:, None, :]

        g_logit = np.einsum("ckj,kj->cj", resp, xi)
        g_m = np.einsum("ckj,ckj,kj->cj", resp, d * inv_s2, xi)
        g_log_s2 = 0.5 * np.einsum("ckj,ckj,kj->cj", resp, d**2 * inv_s2 - 1.0, xi)

        grad_out = np.concatenate([g_logit.T, g_m.T, g_log_s2.T], axis=1)
        # Ascent on Q == descent on -Q; scaling by the edge count keeps the
        # learning rate dataset-size independent.
        grads = self.net.backward(cache, -grad_out / max(stats.n_edges, 1))

        q_before = float(np.sum(stats.xi * self.log_transition_matrix(grid)))
        saved = [p.copy() for p in self.net.params]
        direction = self._adam_direction(grads)
        for scale in self.backtrack_scales:
            for p, s, d_p in zip(self.net.params, saved, direction):
                p[...] = s - scale * d_p
            q_after = float(np.sum(stats.xi * self.log_transition_matrix(grid)))
            if np.isfinite(q_after) and q_after >= q_before:
                return self
        # No ascent found: stay put rather than take a step that would break the
        # guarantee. A run that stalls here is reporting that its learning rate
        # is too large, which is information; a run that silently descends is not.
        for p, s in zip(self.net.params, saved):
            p[...] = s
        return self

    def _adam_direction(self, grads) -> list[np.ndarray]:
        """Adam update direction at unit scale (moments are advanced here)."""
        if self._adam is None:
            self._adam = {
                "m": [np.zeros_like(p) for p in self.net.params],
                "v": [np.zeros_like(p) for p in self.net.params],
                "step": 0,
            }
        st = self._adam
        st["step"] += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        direction = []
        for j, g in enumerate(grads):
            st["m"][j] = b1 * st["m"][j] + (1 - b1) * g
            st["v"][j] = b2 * st["v"][j] + (1 - b2) * g**2
            mh = st["m"][j] / (1 - b1 ** st["step"])
            vh = st["v"][j] / (1 - b2 ** st["step"])
            direction.append(self.lr * mh / (np.sqrt(vh) + eps))
        return direction


def _logsumexp_rows(z: np.ndarray) -> np.ndarray:
    mx = z.max(axis=1)
    return mx + np.log(np.exp(z - mx[:, None]).sum(axis=1))


def _logsumexp_axis0(z: np.ndarray) -> np.ndarray:
    mx = z.max(axis=0)
    return mx + np.log(np.exp(z - mx[None]).sum(axis=0))


def _quad_weights(grid: np.ndarray) -> np.ndarray:
    """Composite trapezoidal weights, matching `bp_grid.make_grid`."""
    dx = float(grid[1] - grid[0])
    w = np.full(len(grid), dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w
