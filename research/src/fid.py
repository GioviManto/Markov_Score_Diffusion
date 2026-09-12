"""Inception Score and Frechet Inception Distance.

Both of these are conventions at least as much as they are statistics, and a
number that does not match the convention is not comparable to anything in the
literature. What follows fixes each convention explicitly and says why.

The pipeline, and the four places it is usually got wrong
---------------------------------------------------------
1. **Which activations.** FID uses `pool3`: the 2048-d global-average-pooled
   output of the last Inception-v3 block, *before* the classifier. IS uses the
   1000-way softmax instead. They are different tensors from the same network.

2. **Which resize.** Images go to 299x299 with **bilinear** interpolation.
   Bicubic or Lanczos shifts FID by more than many reported model differences;
   the resize is part of the metric, not preprocessing before it.

3. **Which weights.** The original numbers come from the TF-Slim Inception graph;
   `torchvision`'s ImageNet weights are a different network with different
   activations. Numbers from the two are *not* comparable, and this is the single
   most common source of FID disagreement between papers. This module records
   which one it used in `ActivationStats.weights` so a stored number carries its
   own provenance. It does not attempt to make them agree -- they do not.

4. **Which n.** See the bias section below. This is the one that silently
   invalidates comparisons rather than merely shifting them.

FID is biased, and the bias is a function of n
----------------------------------------------
FID is a *plug-in* estimator: it fits a Gaussian to each sample and reports the
Frechet distance between the fits. Both the mean and the covariance are estimated
with error, and that error inflates the distance, so E[FID_n] > FID_infinity and
it falls monotonically as n grows. Consequences, in order of how much damage they
do:

* Comparing two models at different n is meaningless. The one with more samples
  wins on arithmetic alone. Fix n -- 50 000 is the convention -- or report
  `bias_curve` and compare the curves.
* Below n = 2048 the sample covariance of a 2048-d activation is **singular**, so
  the estimate is not merely noisy, it is rank-deficient. `fid_from_samples`
  refuses n < 2048 rather than returning a confident-looking number computed from
  a degenerate fit.
* FID(real, real) on two disjoint halves of one dataset is not zero. That value is
  the noise floor of the whole measurement, and no model difference smaller than
  it means anything. `real_vs_real_floor` computes it, and it should be run before
  any model number is believed.

IS is weaker than its ubiquity suggests
---------------------------------------
The Inception Score rewards confident, marginally diverse class predictions. It
cannot see diversity *within* a class, so a model that emits one perfect image per
ImageNet class scores near the ceiling. It also has no access to the reference
data at all -- it never looks at the real images. Report it beside FID or not at
all.

torch
-----
torch is imported lazily, inside the functions that need a network. Everything
statistical here -- the Frechet distance, the bias curve, the validation
harness -- is numpy and scipy, and stays testable on a machine with no torch and
no GPU. That split is deliberate: the parts that can be checked against a closed
form are checked against a closed form, in the ordinary test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

POOL3_DIM = 2048
CONVENTIONAL_N = 50_000


# ----------------------------------------------------------------------------
# The statistic
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationStats:
    """Sufficient statistics of one sample's pool3 activations.

    Stored rather than the activations themselves: the fit only needs a mean and
    a covariance, and 50 000 x 2048 float64 is 800 MB where these are 32 MB.
    `n` travels with them because a FID is not interpretable without it.
    """

    mu: np.ndarray                 # (2048,)
    sigma: np.ndarray              # (2048, 2048)
    n: int
    weights: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_activations(cls, act: np.ndarray, weights: str = "unknown", **meta):
        act = np.asarray(act, dtype=np.float64)
        if act.ndim != 2:
            raise ValueError(f"activations must be (n, d), got {act.shape}")
        return cls(
            mu=act.mean(axis=0),
            # ddof=1. The bias-corrected covariance is what every reference
            # implementation uses (np.cov's default), and at n = 50k the
            # difference is 2e-5 relative -- but it is not zero, and matching the
            # reference is the entire point of this module.
            sigma=np.cov(act, rowvar=False, ddof=1),
            n=act.shape[0],
            weights=weights,
            metadata=meta,
        )


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between two Gaussians.

        d^2 = |mu1 - mu2|^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2})

    `(S1 S2)^{1/2}` is a matrix square root, not an elementwise one, and the
    product of two symmetric PSD matrices is not itself symmetric -- so this needs
    a general `sqrtm`, and `sqrtm` on a near-singular product returns a small
    imaginary part from roundoff. The standard treatment, kept here because
    departing from it would change the number: fall back to a ridge-shifted
    product if the first attempt is non-finite, then assert the imaginary part is
    negligible before discarding it.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("the two samples must have the same dimension")

    diff = mu1 - mu2
    # `sqrtm(A)`, positionally and with no keywords. Every reference FID
    # implementation writes `sqrtm(A, disp=False)` and unpacks a 2-tuple, which
    # is the pre-1.16 scipy signature; `disp` was removed and the return is now
    # the array alone, so that call raises TypeError on a current scipy. The
    # single-argument form has meant the same thing in every version.
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        diag = np.diagonal(covmean).imag
        worst = float(np.max(np.abs(diag))) if diag.size else 0.0
        if worst > 1e-3:
            raise ValueError(
                f"sqrtm returned a substantially complex result (max |Im| on the "
                f"diagonal = {worst:.3e}); the covariances are too ill-conditioned "
                f"for this to be a rounding artefact"
            )
        covmean = covmean.real

    return float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    )


def frechet_decomposition(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> dict:
    """Split FID into its mean term and its covariance (Bures) term.

    FID is the sum of two things that behave completely differently at small n:

        |mu1 - mu2|^2        estimating a d-vector from n samples
        Tr(S1 + S2 - 2(S1 S2)^{1/2})    estimating a d x d matrix from n samples

    and only the first has an exactly known bias. For two independent samples of
    size n from *one* distribution with covariance Sigma,

        E |mu1_hat - mu2_hat|^2 = 2 Tr(Sigma) / n

    exactly, for any distribution with finite covariance -- no Gaussianity
    needed. So the mean term can be predicted rather than merely measured, and
    whatever the measurement exceeds that prediction by is attributable to the
    covariance term.

    This matters for a practical reason. Extrapolating a floor from small n to a
    reporting n is only safe if the thing being extrapolated has a stable law.
    The mean term provably falls as 1/n. The covariance term is estimating d^2/2
    parameters from n samples, and at d = 2048 with n = 10^4 that is the
    marginal regime where sample covariance eigenvalues are badly spread -- so
    its n-dependence is the part an extrapolation is actually betting on.
    Reporting them separately says how much of the bet is on the well-understood
    half.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    diff = mu1 - mu2
    mean_term = float(diff @ diff)
    total = frechet_distance(mu1, sigma1, mu2, sigma2)
    return {
        "fid": total,
        "mean_term": mean_term,
        "covariance_term": total - mean_term,
        "mean_fraction": mean_term / total if total > 0 else float("nan"),
    }


def predicted_mean_term(sigma: np.ndarray, n: int) -> float:
    """`2 Tr(Sigma) / n` -- the exact expected mean term for two size-n samples.

    Exact for any distribution with finite covariance, so it is a prediction the
    measurement can be checked against rather than a Gaussian approximation.
    """
    return float(2.0 * np.trace(np.atleast_2d(sigma)) / n)


def fid_from_stats(a: ActivationStats, b: ActivationStats) -> float:
    """FID between two stored summaries, refusing incomparable pairs.

    Two guards, both for things that produce a plausible number rather than an
    error if left unchecked: mismatched Inception weights, and mismatched n. The
    second is a *warning-level* fact in the literature and a refusal here, because
    the whole reason this module exists is that the mismatched-n comparison is the
    one people keep making.
    """
    if a.weights != b.weights:
        raise ValueError(
            f"activations come from different networks ({a.weights!r} vs "
            f"{b.weights!r}); FIDs from different Inception weights are not "
            f"comparable"
        )
    if a.n != b.n:
        raise ValueError(
            f"sample sizes differ ({a.n} vs {b.n}). FID is biased downwards in n, "
            f"so this comparison would be decided by the sample sizes. Subsample "
            f"the larger to {min(a.n, b.n)}, or compare bias_curve() instead."
        )
    return frechet_distance(a.mu, a.sigma, b.mu, b.sigma)


def fid_from_samples(
    act_a: np.ndarray, act_b: np.ndarray, weights: str = "unknown",
    allow_small_n: bool = False,
) -> float:
    """FID between two activation matrices, with the rank check applied.

    `allow_small_n` exists for tests on low-dimensional synthetic activations,
    where n < 2048 is not a rank problem. It is not for real pool3 features.
    """
    act_a = np.asarray(act_a, dtype=np.float64)
    act_b = np.asarray(act_b, dtype=np.float64)
    dim = act_a.shape[1]
    for name, act in (("a", act_a), ("b", act_b)):
        if not allow_small_n and act.shape[0] <= dim:
            raise ValueError(
                f"sample {name} has n={act.shape[0]} for d={dim}: the sample "
                f"covariance is singular at n <= d, so the Frechet distance would "
                f"be computed from a rank-deficient fit. Use n >= {CONVENTIONAL_N} "
                f"(the convention), or pass allow_small_n=True if this is a "
                f"low-dimensional test."
            )
    return fid_from_stats(
        ActivationStats.from_activations(act_a, weights),
        ActivationStats.from_activations(act_b, weights),
    )


def bias_curve(
    act_a: np.ndarray, act_b: np.ndarray, sizes, n_repeats: int = 3,
    seed: int = 0, allow_small_n: bool = False,
) -> list[dict]:
    """FID against sample size, which is the honest way to report it at small n.

    Returns one row per size with the mean and the spread over `n_repeats`
    independent subsamples. The curve falling as n grows is the bias; a curve that
    has not flattened by the largest size available is a warning that the number
    at that size is still an artefact of n.
    """
    rng = np.random.default_rng(seed)
    act_a = np.asarray(act_a, dtype=np.float64)
    act_b = np.asarray(act_b, dtype=np.float64)
    rows = []
    for n in sizes:
        if n > min(len(act_a), len(act_b)):
            continue
        # At n == len(act) the "subsample" is a permutation of the whole sample,
        # so every repeat draws the *same set* and the spread across repeats is
        # identically zero. That zero is a property of the sizes, not evidence
        # that the estimate is stable -- and it is the largest n, the row a
        # reader is most likely to quote. Say so in the row rather than emitting
        # a 0.0000 that reads as a measurement.
        resampled = n < min(len(act_a), len(act_b))
        reps = n_repeats if resampled else 1
        vals = [
            fid_from_samples(
                act_a[rng.choice(len(act_a), n, replace=False)],
                act_b[rng.choice(len(act_b), n, replace=False)],
                allow_small_n=allow_small_n,
            )
            for _ in range(reps)
        ]
        rows.append({
            "n": int(n),
            "fid_mean": float(np.mean(vals)),
            "fid_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "n_repeats": int(reps),
            # False means the row is a single deterministic evaluation on the
            # whole sample; fid_std is nan, not zero.
            "resampled": bool(resampled),
        })
    return rows


def inception_score(probs: np.ndarray, n_splits: int = 10, seed: int = 0) -> tuple[float, float]:
    """IS = exp(E_x KL(p(y|x) || p(y))), with the conventional 10-way split.

    The split is not a detail: p(y) is estimated *within* each split rather than
    over the whole sample, which changes the number, and the reported standard
    deviation is the spread across splits -- a measure of how unstable the
    estimate is, not a confidence interval for the mean.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (n, n_classes), got {probs.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(probs))
    scores = []
    for part in np.array_split(idx, n_splits):
        p_yx = probs[part]
        p_y = p_yx.mean(axis=0, keepdims=True)
        kl = np.sum(p_yx * (np.log(p_yx + 1e-16) - np.log(p_y + 1e-16)), axis=1)
        scores.append(float(np.exp(np.mean(kl))))
    # ddof=1 over a single split is 0/0. Report 0.0 rather than nan: with one
    # split there is no across-split spread to measure, which is a different
    # statement from "the spread is undefined".
    spread = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return float(np.mean(scores)), spread


# ----------------------------------------------------------------------------
# Validation -- run these before believing any model number
# ----------------------------------------------------------------------------


def real_vs_real_floor(
    real_act: np.ndarray, n_repeats: int = 5, seed: int = 0,
    allow_small_n: bool = False,
) -> dict:
    """FID between two disjoint halves of the real data: the noise floor.

    Small but *not zero*, and the value is the resolution of the whole
    measurement. A model-vs-real FID within this of another model's is not a
    difference between the models. Reported with its spread over repeated splits
    because a single split is itself noisy.
    """
    act = np.asarray(real_act, dtype=np.float64)
    rng = np.random.default_rng(seed)
    half = len(act) // 2
    vals = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(act))
        vals.append(fid_from_samples(
            act[perm[:half]], act[perm[half:2 * half]],
            allow_small_n=allow_small_n,
        ))
    return {
        "floor_mean": float(np.mean(vals)),
        "floor_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_per_half": int(half),
        "n_repeats": int(n_repeats),
    }


def blur_monotonicity(
    fids_by_sigma: dict[float, float], tol: float = 0.0
) -> dict:
    """Check FID rises with the blur radius applied to the generated sample.

    A metric that does not order a sequence of monotonically degraded images
    correctly is misconfigured -- almost always the resize, the channel order, or
    the normalisation. This is a cheap, assumption-free check that catches all
    three, and it is worth running before any model comparison rather than after
    one produces a surprising result.
    """
    sigmas = sorted(fids_by_sigma)
    vals = [fids_by_sigma[s] for s in sigmas]
    violations = [
        {"from_sigma": sigmas[i], "to_sigma": sigmas[i + 1],
         "from_fid": vals[i], "to_fid": vals[i + 1]}
        for i in range(len(vals) - 1)
        if vals[i + 1] < vals[i] - tol
    ]
    return {
        "monotone": not violations,
        "sigmas": sigmas,
        "fids": vals,
        "violations": violations,
    }


# ----------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------


def inception_activations(
    images: np.ndarray,
    batch_size: int = 64,
    device: str | None = None,
    want_probs: bool = False,
):
    """pool3 activations (and optionally softmax probabilities) for a batch.

    `images` is (N, H, W) or (N, H, W, 3) in **[0, 1]**. Greyscale is replicated
    to three channels, which is what this project needs: the wavelet model is
    fitted on CIFAR luminance, and Inception has no single-channel input.

    Conventions applied here, all four of them:
      * bilinear resize to 299x299, `align_corners=False`;
      * torchvision ImageNet weights, recorded in the returned `weights` string
        so the number is never silently compared against a TF-Slim one;
      * pool3 = the 2048-d output of `avgpool`, taken by replacing `fc` with an
        identity rather than by re-running a truncated graph;
      * ImageNet mean/std normalisation, which the torchvision weights expect and
        the TF-Slim graph does not.

    Returns `(ActivationStats, probs_or_None)`.
    """
    import torch                      # lazy: see the module docstring
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3

    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images must be (N,H,W) or (N,H,W,3), got {arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -1e-3 or hi > 1.0 + 1e-3:
        raise ValueError(
            f"images must be in [0, 1]; got [{lo:.3f}, {hi:.3f}]. Rescaling here "
            f"would hide a normalisation bug that changes every number below."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=weights, transform_input=False).to(device).eval()
    head = net.fc
    net.fc = torch.nn.Identity()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats, probs = [], []
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch = torch.from_numpy(arr[start:start + batch_size]).to(device)
            batch = batch.permute(0, 3, 1, 2)                 # NHWC -> NCHW
            batch = F.interpolate(
                batch, size=(299, 299), mode="bilinear", align_corners=False,
            )
            batch = (batch - mean) / std
            f = net(batch)
            feats.append(f.cpu().numpy())
            if want_probs:
                probs.append(torch.softmax(head(f), dim=1).cpu().numpy())

    act = np.concatenate(feats, axis=0)
    stats = ActivationStats.from_activations(
        act, weights="torchvision:Inception_V3_Weights.IMAGENET1K_V1",
        device=device, n_images=len(arr),
    )
    return stats, (np.concatenate(probs, axis=0) if want_probs else None), act
