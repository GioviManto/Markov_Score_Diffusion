"""A transition kernel whose conditional *scale* depends on the parent.

Why this kernel exists
----------------------
Every kernel in `src/kernels.py` is linear-autoregressive,

    K(a' | a) = phi(a' - rho a),

so the parent's entire influence on the child is a shift of the innovation's
*location*: the conditional variance does not depend on `a` at all. On a chain of
simulated AR(1) data that is the right family, because it is the generating one.

On natural-image wavelet coefficients it is the wrong family, and
`experiments/exp_23_wavelet_statistics.py` measures by how much. Across the
finest scale boundary of CIFAR-10, a top-quartile-magnitude parent has children
2.9-3.6x more spread than a bottom-quartile one, in every orientation. And the
HH subbands, where that magnitude effect is strongest, have almost no *linear*
parent-child correlation (0.04-0.15). Fitted with a linear-AR kernel, HH would
return rho ~ 0 and collapse to a factorised heavy-tailed model: good marginals,
no hierarchy at all.

This is the continuous-density form of the classical wavelet hidden Markov tree
(Crouse, Nowak, Baraniuk 1998), whose hidden state is exactly a "this
coefficient is large / small" indicator passed from parent to child. There the
tree is Markov in the hidden *state*; here it is Markov in the value, which is
what the package's grid BP requires, so the state is marginalised into the gate:

    K(a' | a) = sum_c w_c(a) N(a'; rho_c a, sigma_c^2),
    w_c(a)    = softmax_c(beta_c + gamma_c a^2).

The quadratic-in-`a` logit is not an arbitrary choice: `softmax_c(beta_c +
gamma_c a^2)` with `gamma_c = -1/(2 tau_c^2)` and `beta_c = log pi_c - 0.5 log
tau_c^2` *is* the posterior responsibility of a zero-mean Gaussian scale mixture.
So the gate is the exact form the scale-mixture story implies, with `C` mixture
components costing `4C` parameters per level.

The M-step
----------
ECM with the component label as the inner latent variable, exactly as
`MixtureInnovationKernel` does. Given the responsibilities

    R_c(k, j) = w_c(u_j) N(u_k; rho_c u_j, sigma_c^2) / K(u_k | u_j),

the `rho_c` and `sigma_c^2` blocks are closed-form weighted least squares. The
gate block is *not* closed form: maximising `sum_{j,c} n_c(j) log w_c(u_j)` is a
multinomial logistic regression on the feature `(1, u_j^2)` with soft counts. It
is concave, so gradient ascent with a backtracking line search reaches an
increase every time, which is all generalised EM needs -- the same ECM framing
the project already uses, and monotone ascent is preserved rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .em import ExpectedStatistics

_LOG_2PI = float(np.log(2.0 * np.pi))
_VAR_FLOOR = 1e-6


# Variance of a standard normal restricted to the bottom and top quartiles of
# |a| (cut points 0.3186 and 1.1503). Used to say what a purely linear model
# would already produce on the magnitude statistic.
_VAR_Q1 = 0.03360
_VAR_Q4 = 2.89424


def linear_ar_magnitude_ratio(rho: float) -> float:
    """The `std_ratio_q4_q1` a *purely linear* AR model would already produce.

    This is the null the empirical statistic has to be read against, and leaving
    it out overstates the case. Conditioning the child on a *set* of parent
    values -- which is what the quartile statistic does -- picks up the spread of
    the conditional mean `rho a` across that set, not only any dependence of the
    conditional *variance* on `a`. For a homoscedastic AR(1) with unit marginal
    variance,

        Var(child | a in S) = rho^2 Var(a | S) + (1 - rho^2),

    so the ratio exceeds 1 whenever rho does not vanish: 1.31 at rho = 0.45,
    1.61 at rho = 0.60. Agreement with simulation is to three decimals
    (`tests/test_scale_kernel.py`).
    """
    r2 = float(rho) ** 2
    return float(np.sqrt((r2 * _VAR_Q4 + 1.0 - r2) / (r2 * _VAR_Q1 + 1.0 - r2)))


def magnitude_diagnostics(kernel, grid: np.ndarray) -> dict:
    """`std_ratio_q4_q1`, its linear-AR null, and the excess, for any kernel.

    This is the measurement that decides whether the scale-mixture family did the
    job it was built for, so it is computed the same way for every family rather
    than only for the one expected to win. `exp_23` reports an empirical excess
    of 1.86 (HH) to 2.33 (HL) at the finest scale boundary; a fitted kernel that
    reproduces that has captured the structure, and one that returns an excess
    near 1 has not -- whatever its held-out likelihood says.

    The excess is `ratio / null` with the null evaluated at the kernel's *own*
    implied slope, matching how the empirical number is read in the handover.

    For a linear-AR kernel the two agree by construction and the excess is
    exactly 1. That is a consistency check, not a wasted computation: it runs on
    the gaussian and mixture arms of every fit and would catch a quadrature or
    convention error in the diagnostic itself before it was used to make a claim
    about the scale mixture.
    """
    if hasattr(kernel, "magnitude_ratio"):
        rho_eff = float(kernel.implied_rho(grid))
        ratio = float(kernel.magnitude_ratio(grid))
    else:
        # Linear-AR families: scalar rho, conditional variance independent of the
        # parent, so the closed form *is* the model's prediction.
        rho_eff = float(np.asarray(getattr(kernel, "rho", np.nan)).reshape(-1)[0])
        ratio = linear_ar_magnitude_ratio(rho_eff)
    null = linear_ar_magnitude_ratio(rho_eff)
    return {
        "rho_implied": rho_eff,
        "magnitude_ratio": ratio,
        "magnitude_null": null,
        "magnitude_excess": ratio / null if null > 0 else float("nan"),
    }


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class ScaleMixtureKernel:
    """K(a'|a) = sum_c softmax_c(beta + gamma a^2) N(a'; rho_c a, s2_c).

    `beta[0]` and `gamma[0]` are held at zero: the softmax is invariant to a
    common shift of the logits, so without a reference class the parameters
    drift along a flat direction and the fitted values stop being comparable
    across levels.
    """

    rho: np.ndarray      # (C,)
    s2: np.ndarray       # (C,)
    beta: np.ndarray     # (C,), beta[0] == 0
    gamma: np.ndarray    # (C,), gamma[0] == 0
    gate_steps: int = 12

    @property
    def name(self) -> str:
        return f"scale_mixture_C{len(self.rho)}"

    @property
    def n_components(self) -> int:
        return len(self.rho)

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.rho, self.s2, self.beta, self.gamma])

    @classmethod
    def init(
        cls, n_components: int, rho: float, var: float, rng: np.random.Generator
    ) -> "ScaleMixtureKernel":
        """Components spread over a range of scales, all sharing the initial rho.

        Spreading the *scales* rather than the locations is the point: the
        components have to be able to specialise into "this child is small" and
        "this child is large" before the gate has anything to select between.
        """
        c = n_components
        spread = np.geomspace(0.35, 2.2, c) if c > 1 else np.ones(1)
        s2 = np.maximum(var * spread**2, _VAR_FLOOR)
        beta = np.zeros(c)
        gamma = np.zeros(c)
        if c > 1:
            # Break the tie so the gate has a gradient at iteration one; small
            # enough that the first E-step is still close to an equal mixture.
            gamma[1:] = np.linspace(0.05, 0.25, c - 1) * rng.uniform(0.8, 1.2, c - 1)
        return cls(
            rho=np.full(c, float(rho)), s2=s2, beta=beta, gamma=gamma,
        )

    # -- density -----------------------------------------------------------

    def _gate_logits(self, grid: np.ndarray) -> np.ndarray:
        """(M, C) logits as a function of the *parent* value u_j."""
        return self.beta[None, :] + self.gamma[None, :] * (grid**2)[:, None]

    def gate(self, grid: np.ndarray) -> np.ndarray:
        """(M, C) mixing weights w_c(u_j), rows summing to 1."""
        return _softmax_rows(self._gate_logits(grid))

    def _component_logs(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        """(C, M_out, M_in) log[ w_c(u_j) N(u_k; rho_c u_j, s2_c) ], [c, k, j].

        The gate reads only the *parent* value, so it lives on `grid` whatever
        the child grid is; only the residual spans both.
        """
        out = grid if grid_out is None else grid_out
        w = self.gate(grid)                                   # (M_in, C)
        resid = out[None, :, None] - self.rho[:, None, None] * grid[None, None, :]
        s2 = self.s2[:, None, None]
        return (
            np.log(np.maximum(w.T, 1e-300))[:, None, :]
            - 0.5 * (_LOG_2PI + np.log(s2))
            - 0.5 * resid**2 / s2
        )

    def log_transition_matrix(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        from scipy.special import logsumexp

        return logsumexp(self._component_logs(grid, grid_out), axis=0)

    def responsibilities(
        self, grid: np.ndarray, grid_out: np.ndarray | None = None
    ) -> np.ndarray:
        from scipy.special import logsumexp

        comp = self._component_logs(grid, grid_out)
        return np.exp(comp - logsumexp(comp, axis=0)[None, :, :])

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ScaleMixtureKernel has no gradient interface: its M-step is ECM "
            "(closed form for rho/s2, line-searched ascent for the gate). The "
            "Fisher-identity experiment (exp_08) applies to the chain kernels."
        )

    # -- diagnostics -------------------------------------------------------

    def conditional_std(self, grid: np.ndarray) -> np.ndarray:
        """(M,) std of the child given each parent value -- the fitted analogue of
        `std_ratio_q4_q1` in `exp_23`, and the number that says whether the
        magnitude dependence was actually learned."""
        w = self.gate(grid)                                   # (M, C)
        mean = w @ self.rho * grid
        second = (w * (self.s2[None, :] + (self.rho[None, :] * grid[:, None]) ** 2)).sum(1)
        return np.sqrt(np.maximum(second - mean**2, 0.0))

    def implied_rho(self, grid: np.ndarray) -> float:
        """The lag-1 correlation a *linear* AR fit would return for this kernel.

        This is the number that makes the HH case in the cross-scale measurement
        legible. A least-squares fit of child on parent returns
        Cov(a', a) / Var(a), which for a standard-normal parent is just
        E[a a'] = E[a * E[a'|a]]. Under this kernel the conditional mean is
        `rho_bar(a) * a` with `rho_bar(a) = sum_c w_c(a) rho_c`, so the slope
        averages the components *weighted by where the gate puts them*.

        A kernel can therefore carry strong magnitude dependence and still report
        a slope near zero -- which is exactly what a linear-AR family sees on HH,
        and why it collapses to independence there while the real structure is
        untouched. Reporting this beside `magnitude_ratio` says how much of the
        dependence a linear model was ever able to see.

        Same quadrature convention as `magnitude_ratio`: a standard-normal parent
        on the given (uniform) grid.
        """
        w = self.gate(grid)                                   # (M, C)
        cond_mean = (w @ self.rho) * grid
        dens = np.exp(-0.5 * grid**2)
        p = dens / dens.sum()
        return float(p @ (grid * cond_mean))

    def magnitude_ratio(self, grid: np.ndarray) -> float:
        """Model analogue of `exp_23`'s empirical `std_ratio_q4_q1`.

        Comparable to the sample statistic, but not exactly: see the caveat
        below. The empirical version takes the standard deviation of all children
        whose parent lies in the top quartile of |a|, over the same for the
        bottom quartile. So this integrates the conditional law over those same
        two sets under a standard-normal parent, using the variance
        decomposition

            Var(child | a in S) = E[Var(child|a) | S] + Var(E[child|a] | S),

        and not the conditional standard deviation evaluated at a single
        representative `a`. The two differ substantially, because the top
        quartile reaches far into the tail where the spread is largest, and
        quoting the pointwise version against the empirical one would understate
        the fitted dependence.

        Quartiles of |a| for a standard normal: 0.3186 and 1.1503.

        THE CAVEAT, measured 21 Aug 2026. `wavelet_stats` cuts the empirical
        bands at the EMPIRICAL quantiles of |a|, and standardised wavelet
        coefficients are far from normal -- excess kurtosis rises from ~0.2 at
        the coarsest subband to ~7 at the finest, so unit variance is reached
        with a compressed bulk and heavy tails. The real cuts at depth 4 are
        about 0.09 and 0.78, against 0.3186 and 1.1503 here: the Q1 band assumed
        by this function is nearly four times too wide, which carries extra
        spread into the denominator and DEPRESSES the ratio.

        The bias was quantified against the fitted kernels by re-integrating on
        the empirical bands with the empirical parent density: it is under 1% at
        depth 2 and up to 8% at depth 3 (HL 1.183 -> 1.270). So it is real, it
        runs in the direction that flatters a linear model, and it is small. It
        does NOT explain the gap between fitted excess (1.15-1.31 at the finest
        boundary) and measured (1.86-2.33) -- that gap survives the correction
        and remains open.

        Left as the normal-parent version because this function has no access to
        the data, and a diagnostic whose value depends on which sample it was
        handed is worse than one with a stated convention. Quote it against the
        empirical statistic only with the above in mind.
        """
        w = self.gate(grid)                                   # (M, C)
        cond_mean = (w @ self.rho) * grid
        cond_var = (
            (w * (self.s2[None, :] + (self.rho[None, :] * grid[:, None]) ** 2)).sum(1)
            - cond_mean**2
        )
        dens = np.exp(-0.5 * grid**2)

        def band_std(mask):
            p = dens * mask
            total = p.sum()
            if total <= 0:
                return float("nan")
            p = p / total
            mean = float(p @ cond_mean)
            return float(np.sqrt(p @ (cond_var + cond_mean**2) - mean**2))

        lo = band_std(np.abs(grid) <= 0.3186)
        hi = band_std(np.abs(grid) >= 1.1503)
        return hi / lo if lo > 0 else float("nan")

    # -- M-step ------------------------------------------------------------

    def m_step(
        self, stats: ExpectedStatistics, grid: np.ndarray,
        grid_out: np.ndarray | None = None, n_inner: int = 3,
    ) -> "ScaleMixtureKernel":
        xi = stats.xi
        out = grid if grid_out is None else grid_out
        current = self
        for _ in range(n_inner):
            r = current.responsibilities(grid, out)            # (C, M_k, M_j)
            w = r * xi[None, :, :]

            mass = np.maximum(w.sum(axis=(1, 2)), 1e-300)
            # rho_c: weighted least squares of child on parent, per component.
            num = (w * grid[None, None, :] * out[None, :, None]).sum(axis=(1, 2))
            den = np.maximum((w * (grid**2)[None, None, :]).sum(axis=(1, 2)), 1e-300)
            rho = num / den
            resid = out[None, :, None] - rho[:, None, None] * grid[None, None, :]
            s2 = np.maximum((w * resid**2).sum(axis=(1, 2)) / mass, _VAR_FLOOR)
            current = replace(current, rho=rho, s2=s2)

            # Gate block: soft counts per parent value, then concave ascent.
            # Summing over the child axis is what makes this independent of the
            # child grid -- the gate is a function of the parent alone.
            r = current.responsibilities(grid, out)
            counts = (r * xi[None, :, :]).sum(axis=1).T        # (M_j, C)
            current = current._fit_gate(grid, counts)
        return current

    def _fit_gate(self, grid: np.ndarray, counts: np.ndarray) -> "ScaleMixtureKernel":
        """Multinomial logistic ascent on the gate, with a backtracking line search.

        Concave, so any step that increases the objective is safe for generalised
        EM. The line search is what turns "gradient ascent" into a guarantee
        rather than a hope: if no tried step improves, the parameters are left
        untouched and the EM iteration is still monotone.
        """
        feat = np.stack([np.ones_like(grid), grid**2], axis=1)   # (M, 2)
        total = counts.sum()
        if total <= 0:
            return self

        def objective(beta, gamma):
            logits = beta[None, :] + gamma[None, :] * (grid**2)[:, None]
            logits = logits - logits.max(axis=1, keepdims=True)
            log_w = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            return float((counts * log_w).sum())

        beta, gamma = self.beta.copy(), self.gamma.copy()
        obj = objective(beta, gamma)
        step = 1.0 / max(total, 1.0)

        for _ in range(self.gate_steps):
            logits = beta[None, :] + gamma[None, :] * (grid**2)[:, None]
            w = _softmax_rows(logits)                            # (M, C)
            resid = counts - w * counts.sum(axis=1, keepdims=True)
            grad = feat.T @ resid                                # (2, C)
            # Reference class fixed at zero; never move it.
            grad[:, 0] = 0.0
            if not np.all(np.isfinite(grad)):
                break

            scale = step
            improved = False
            for _ in range(20):
                nb = beta + scale * grad[0]
                ng = gamma + scale * grad[1]
                nb[0] = 0.0
                ng[0] = 0.0
                new_obj = objective(nb, ng)
                if np.isfinite(new_obj) and new_obj > obj:
                    beta, gamma, obj = nb, ng, new_obj
                    improved = True
                    step = scale * 1.5
                    break
                scale *= 0.5
            if not improved:
                break

        return replace(self, beta=beta, gamma=gamma)
