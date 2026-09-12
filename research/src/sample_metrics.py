"""Metrics for comparing a *generated* distribution against the true one.

Why this module exists
----------------------
Every comparison in the project so far has been *pointwise*: given the same noisy
observation ``x``, how far is one denoiser from another. Jerome's objection in the call of
29 July is that this is not the quantity diffusion is judged on::

    "the error on the score, the relationship between the score and the distribution of
     data that you end up generating at the end of the diffusion process, is not trivial,
     because you accumulate the errors of the scores, you go backwards, and not all times
     are equal."

A denoiser is used inside a reverse integration that calls it hundreds of times. Errors
compound, and they compound unevenly across the schedule: at small ``t`` the denoiser barely
acts (``alpha_t ~ 1``), at large ``t`` the noise dominates whatever it does, so it is the
intermediate times that govern the generated distribution. A pointwise error averaged over
``t`` cannot see any of that.

So this module measures the *output distribution*: draw samples by running the reverse
process under each score, and ask how the resulting law differs from ``p_0``.

The four metric families, and why all four
------------------------------------------
The project reports four things, and the reason is that they answer different questions and
have historically disagreed:

1. ``MSE(m_hat, m_star)`` -- distance to the Bayes-optimal denoiser. **Pure method error**:
   zero for a perfect method, no floor. This is the primary pointwise metric, and the one
   Jerome meant by "the posterior mean is the better metric".

2. ``MSE(m_hat, a)`` -- the actual denoising risk, i.e. the training loss. **Has an
   irreducible floor** at the Bayes risk ``E||a - E[a|x]||^2``. Reporting it without the
   floor is meaningless and has already produced one unsatisfiable test criterion in this
   project (a demand for >40% loss reduction when the arithmetic maximum was 39%).

3. Relative score error and cosine -- kept for continuity with Layers 2-5. **De-emphasised**,
   for a precise reason: by Tweedie,

       ||s_hat - s_star||  =  (alpha_t / delta_t) * ||m_hat - m_star||

   exactly (see ``metrics.score_mean_identity_residual``, which pins this to machine
   precision). So (1) and (3) are the *same measurement* under a known, strongly
   ``t``-dependent reweighting: ``alpha/delta`` runs from ~6.0 at ``t=0.08`` to ~0.09 at
   ``t=2.4``, a factor of 65 across the schedule. Score error silently upweights low noise.
   Both are reported so a reader can convert; neither is presented alone.

4. **Distributional metrics on generated samples** -- this module. What was actually asked
   for, and the only family that sees error accumulation.

Scope and honesty
-----------------
These are two-sample statistics computed from finite samples, so every one of them carries
Monte Carlo error, and a difference smaller than that error is not a difference. Each
function here therefore returns a bootstrap standard error alongside the estimate, and
``compare_distributions`` refuses to declare a winner when the intervals overlap. That is
deliberate: the project has twice reported a structure that a replicate count later
dissolved (the N^{-1/2} rate at 4 replicates, the kurtosis "convergence curve" at one
replicate per N).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Innovation recovery
# ---------------------------------------------------------------------------

def ar_residuals(a: np.ndarray, rho: float) -> np.ndarray:
    """AR-filtered residuals ``r_i = a_i - rho a_{i-1}`` of a sequence.

    ``a`` has shape ``(n_chains, n_sites)``; the returned array has ``n_sites - 1`` columns
    because ``r_1`` is not identified without knowing the initial draw.

    **These are innovations only when ``a`` is the clean chain.** Every generated sample in
    this project is a draw from ``p_{t_min}``, not from ``p_0`` -- the reverse integration
    must stop at ``t_min > 0`` -- and there the filter does not recover the innovations. With
    ``x_i = alpha a_i + sqrt(Delta) z_i`` and ``a_i = rho a_{i-1} + eps_i``,

        r_i = alpha eps_i + sqrt(Delta) (z_i - rho z_{i-1}),

    so adjacent residuals are correlated:

        Cov(r_i, r_{i+1}) = -rho Delta,
        Corr(r_i, r_{i+1}) = -rho Delta / (alpha^2 q + Delta (1 + rho^2)).

    The noised process is not AR(1) at all -- even in the Gaussian case it is ARMA-type --
    so calling ``r`` an innovation overstates what has been recovered. The *marginal* shape
    comparison remains valid and is still the sharpest available discriminator between
    families, since the covariance is matched by construction and only shape distinguishes
    them. What does not survive is the claim that this measures recovery of the clean
    transition innovation law.

    Named `ar_residuals` for that reason; `innovations` is kept as a deprecated alias so no
    committed caller breaks silently, and `residual_autocorrelation` below reports the
    correlation this docstring predicts, so the caveat is measured rather than asserted.
    """
    if a.ndim != 2:
        raise ValueError(f"expected (n_chains, n_sites), got shape {a.shape}")
    return a[:, 1:] - rho * a[:, :-1]


#: Deprecated alias. The quantity is an AR-filtered residual, not an innovation, whenever it
#: is applied to anything other than a clean chain -- which in this project is always.
innovations = ar_residuals


def residual_autocorrelation(r: np.ndarray, max_lag: int = 5) -> list[float]:
    """Autocorrelation of the AR-filtered residuals, lags 1..max_lag.

    Zero at every lag exactly when ``r`` really is an innovation sequence. On a sample of
    ``p_{t_min}`` lag 1 is predicted to be ``-rho Delta / (alpha^2 q + Delta (1 + rho^2))``,
    so this is the diagnostic that turns the caveat above into a number a reader can check
    against the closed form.
    """
    if r.ndim != 2:
        raise ValueError(f"expected (n_chains, n_lags), got shape {r.shape}")
    c = r - r.mean()
    var = float((c**2).mean())
    if var <= 0.0:
        return [float("nan")] * max_lag
    out = []
    for lag in range(1, max_lag + 1):
        if c.shape[1] <= lag:
            out.append(float("nan"))
        else:
            out.append(float((c[:, :-lag] * c[:, lag:]).mean() / var))
    return out


def predicted_residual_autocorr(rho: float, q: float, t_min: float) -> float:
    """Closed-form lag-1 residual correlation of ``p_{t_min}``.

    ``-rho Delta / (alpha^2 q + Delta (1 + rho^2))``, the quantity `residual_autocorrelation`
    should reproduce. Having both means the "these are not innovations" correction is
    verifiable rather than a footnote.
    """
    alpha = float(np.exp(-t_min))
    delta = float(1.0 - np.exp(-2.0 * t_min))
    return -rho * delta / (alpha**2 * q + delta * (1.0 + rho**2))


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for a Gaussian), pooled over all entries.

    Uses the plug-in estimator rather than the bias-corrected one: with the sample sizes
    here (>= 10^4 entries) the difference is far below the sampling error reported by
    ``bootstrap_se``, and the plug-in version is what the true values quoted in the project
    (Laplace 3.0, uniform -1.2, Student-t(5) 6.0) refer to.
    """
    v = np.asarray(x).ravel()
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float("nan")
    c = v - v.mean()
    m2 = float(np.mean(c ** 2))
    if m2 <= 0.0:
        return float("nan")
    return float(np.mean(c ** 4) / m2 ** 2 - 3.0)


def bootstrap_se(
    x: np.ndarray,
    statistic,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """Bootstrap standard error of ``statistic`` over the *rows* of ``x``.

    Resampling is over chains, not over entries, because entries within a chain are
    correlated by construction -- that is the entire point of the data model. Bootstrapping
    entries would understate the error by roughly the correlation length.
    """
    a = np.asarray(x)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    if n < 2:
        return float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = statistic(a[idx])
    return float(np.std(vals, ddof=1))


# ---------------------------------------------------------------------------
# Distributional distances
# ---------------------------------------------------------------------------

def histogram_kl(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int = 60,
    range_quantile: float = 0.999,
) -> float:
    """Estimate ``KL(p || q)`` from samples by binning both on a shared grid.

    ``p_samples`` is the *reference* (true) sample and ``q_samples`` the generated one, so
    this is the divergence of the truth from what the model produced -- the direction that
    penalises a model for putting no mass where the truth has some, which is exactly the
    heavy-tail failure this experiment is looking for.

    Caveats, stated because a binned KL is easy to misread:

    * It is biased upward and the bias grows with ``n_bins``. Comparisons across methods are
      valid only at *fixed* binning, which is why the grid is derived from the reference
      sample alone and never from the sample under test.
    * Bins where the reference has no mass contribute nothing; bins where the reference has
      mass and the model has none would give infinity, so ``q`` is floored at one
      pseudo-count. The floor makes the number finite but means a catastrophic model is
      reported as merely bad. Read it alongside the moment statistics, not instead of them.
    """
    p = np.asarray(p_samples).ravel()
    q = np.asarray(q_samples).ravel()
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if p.size == 0 or q.size == 0:
        return float("nan")

    lo, hi = np.quantile(p, [1.0 - range_quantile, range_quantile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, n_bins + 1)

    cp, _ = np.histogram(p, bins=edges)
    cq, _ = np.histogram(q, bins=edges)
    # One pseudo-count in q only: p defines the support, q must not be allowed a hard zero.
    pp = cp / max(cp.sum(), 1)
    qq = (cq + 1.0) / (cq.sum() + n_bins)

    mask = pp > 0
    return float(np.sum(pp[mask] * np.log(pp[mask] / qq[mask])))


def covariance_error(a_gen: np.ndarray, sigma_true: np.ndarray) -> tuple[float, float]:
    """Return ``(frobenius_rel_error, worst_lag_abs_error)`` of the sample covariance.

    The second number is the more interpretable one for a chain: it is the largest absolute
    deviation of the empirical ``rho^|i-j|`` profile, averaged over the diagonal band at each
    lag, so it says *at which separation* the generated data stops matching.
    """
    g = np.asarray(a_gen)
    s = np.cov(g, rowvar=False)
    denom = float(np.linalg.norm(sigma_true, "fro"))
    fro = float(np.linalg.norm(s - sigma_true, "fro")) / denom if denom > 0 else np.nan

    n = s.shape[0]
    worst = 0.0
    for lag in range(n):
        d_est = float(np.mean(np.diag(s, lag)))
        d_ref = float(np.mean(np.diag(sigma_true, lag)))
        worst = max(worst, abs(d_est - d_ref))
    return fro, worst


# ---------------------------------------------------------------------------
# The reported bundle
# ---------------------------------------------------------------------------

@dataclass
class SampleComparison:
    """Everything measured about one generated sample, against one reference."""

    name: str
    n_generated: int
    innov_kurtosis: float
    innov_kurtosis_se: float
    innov_kurtosis_true: float
    innov_variance: float
    innov_variance_true: float
    innov_kl: float
    cov_frobenius_rel: float
    cov_worst_lag_abs: float
    marginal_mean: float
    marginal_var: float
    notes: list[str] = field(default_factory=list)

    def kurtosis_gap_in_se(self) -> float:
        """How many standard errors the kurtosis sits from the truth.

        This is the headline number for the heavy-tail question, and expressing it in
        standard errors rather than absolute terms is what stops a small sample from looking
        like a finding. Below ~2 the sample does not distinguish the model from the truth.
        """
        if not np.isfinite(self.innov_kurtosis_se) or self.innov_kurtosis_se <= 0:
            return float("nan")
        return abs(self.innov_kurtosis - self.innov_kurtosis_true) / self.innov_kurtosis_se


def compare_distributions(
    a_gen: np.ndarray,
    a_ref: np.ndarray,
    rho: float,
    sigma_true: np.ndarray,
    innov_kurtosis_true: float,
    innov_variance_true: float,
    name: str = "",
    seed: int = 0,
) -> SampleComparison:
    """Full distributional comparison of a generated sample against a reference sample.

    ``a_ref`` should be drawn from the *true* prior by the forward sampler, not by any
    reverse process -- it is the yardstick, and using a reverse-generated reference would
    fold the integrator's own error into the target.
    """
    e_gen = innovations(a_gen, rho)
    e_ref = innovations(a_ref, rho)

    k = excess_kurtosis(e_gen)
    k_se = bootstrap_se(e_gen, excess_kurtosis, seed=seed)

    fro, worst = covariance_error(a_gen, sigma_true)

    notes: list[str] = []
    if a_gen.shape[0] < 500:
        notes.append(
            f"only {a_gen.shape[0]} generated chains; kurtosis SE is large and the "
            "comparison is weak"
        )
    if not np.all(np.isfinite(a_gen)):
        n_bad = int(np.sum(~np.isfinite(a_gen)))
        notes.append(f"{n_bad} non-finite entries in the generated sample (integrator blew up)")

    return SampleComparison(
        name=name,
        n_generated=int(a_gen.shape[0]),
        innov_kurtosis=k,
        innov_kurtosis_se=k_se,
        innov_kurtosis_true=float(innov_kurtosis_true),
        innov_variance=float(np.var(e_gen)),
        innov_variance_true=float(innov_variance_true),
        innov_kl=histogram_kl(e_ref, e_gen),
        cov_frobenius_rel=fro,
        cov_worst_lag_abs=worst,
        marginal_mean=float(np.mean(a_gen)),
        marginal_var=float(np.var(a_gen)),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# The pointwise ladder, with its floor
# ---------------------------------------------------------------------------

def bayes_risk(m_star: np.ndarray, a: np.ndarray) -> float:
    """The irreducible floor of the denoising loss, ``E||a - E[a|x]||^2`` per site.

    Any statement about ``MSE(m_hat, a)`` is uninterpretable without this: the loss cannot
    go below it, so "how much of the achievable reduction was captured" is the only
    meaningful normalisation. ``m_star`` must be the *exact* posterior mean; passing an
    approximation here silently inflates the floor and flatters every method measured
    against it.
    """
    return float(np.mean((a - m_star) ** 2))


def pointwise_ladder(
    m_hat: np.ndarray,
    m_star: np.ndarray,
    a: np.ndarray,
    alpha: float,
    delta: float,
) -> dict[str, float]:
    """All pointwise metrics for one denoiser at one noise level.

    Returns method error, denoising risk, the Bayes floor, the fraction of the achievable
    reduction captured, and the score-space equivalents -- which are related to the first by
    the exact factor ``alpha/delta`` recorded in ``score_reweighting``. Reporting that factor
    alongside the numbers is what lets a reader see *why* the score ranking differs from the
    posterior-mean ranking, rather than having to rediscover it.
    """
    method_mse = float(np.mean((m_hat - m_star) ** 2))
    risk = float(np.mean((m_hat - a) ** 2))
    floor = bayes_risk(m_star, a)
    ceiling = float(np.mean(a ** 2))  # the trivial predictor m = 0

    achievable = ceiling - floor
    captured = (ceiling - risk) / achievable if achievable > 0 else float("nan")

    denom = float(np.linalg.norm(m_star))
    rel_mean = float(np.linalg.norm(m_hat - m_star)) / denom if denom > 0 else float("nan")

    return {
        "mse_vs_bayes_denoiser": method_mse,
        "rel_l2_vs_bayes_denoiser": rel_mean,
        "denoising_risk": risk,
        "bayes_risk_floor": floor,
        "zero_predictor_ceiling": ceiling,
        "fraction_achievable_captured": captured,
        "score_reweighting": alpha / delta if delta > 0 else float("inf"),
        # By the Tweedie identity this equals rel_l2 in score space up to ||s*|| vs ||m*||;
        # both are emitted so the conversion is visible rather than asserted.
        "abs_score_error_implied": (alpha / delta) * float(np.linalg.norm(m_hat - m_star)),
    }
