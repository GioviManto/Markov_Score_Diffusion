"""Experiment 30 -- validate the FID implementation before it scores anything.

This runs no model. It is a gate on the *metric*, and it exists because a
misconfigured FID does not fail: it returns a plausible number, in the range
readers expect, that happens to rank models wrongly. The three failure modes that
produce plausible-but-wrong numbers are the resize filter, the channel handling
and the normalisation, and all three are invisible in the output.

So the metric is checked against things whose answer is known in advance:

floor     FID between two disjoint halves of the *real* data. Not zero -- it is
          the sampling noise of a 2048-d covariance estimated from n images --
          and its value is the resolution of every comparison made afterwards.
          A model difference smaller than the floor is not a difference.

blur      FID of the real data against a progressively Gaussian-blurred copy of
          itself. This must increase with the blur radius. It is assumption-free:
          blurring destroys information, so any correctly wired perceptual metric
          orders the sequence. A flat or non-monotone curve means the pipeline is
          wrong, and says so before any model is involved.

bias      FID against sample size, on two disjoint halves of the real data. The
          curve must fall, because FID is a plug-in estimator of a distance
          between *fitted* Gaussians and the fit error inflates the distance.
          Its purpose is to make the size of that effect concrete for this
          dataset, so that a later model comparison at some n can be checked
          against the bias at that same n rather than assumed to be free of it.

Run all three and read them together before quoting any FID. If the floor is
large relative to the model differences of interest, the answer is more samples,
not a tighter argument.

This uses `.venv-metrics`, not the package venv: torch's CUDA wheels are kept out
of the environment the BP results are computed in, because that environment has a
load-bearing `cupy < 14` pin and a CUDA 12.4 path that a second CUDA stack could
disturb. Nothing here imports `src.wavelet*`, so the split costs nothing.

    .venv-metrics/bin/python experiments/exp_30_fid_validation.py \
        --output-dir outputs/exp_30_fid_validation
"""

from __future__ import annotations

import numpy as np

from common import (
    apply_overrides, experiment_parser, provenance, select_parts,
)
from src.fid import (
    ActivationStats, CONVENTIONAL_N, bias_curve, blur_monotonicity,
    fid_from_samples, frechet_decomposition, inception_activations,
    predicted_mean_term, real_vs_real_floor,
)
from src.image_data import load_cifar10_luminance
from src.utils import ensure_dir, write_csv, write_json

SETTINGS = {
    "archive": "data/cifar-10-python.tar.gz",
    # 20 000 real images: 10 000 per half, which is comfortably above the 2048-d
    # rank floor while staying inside one debug-queue GPU slot. The conventional
    # 50 000 is what a *published* number needs; this is a validation run, and
    # what it has to establish is the shape of the curves, not a citable value.
    "n_images": 20_000,
    "batch_size": 128,
    "blur_sigmas": (0.0, 0.5, 1.0, 2.0, 4.0),
    "bias_sizes": (2500, 5000, 10_000),
    "n_repeats": 4,
    "seed": 0,
    # Only --quick sets this. pool3 is 2048-d, so a smoke run's few hundred
    # images are below the rank floor and every FID call would (correctly)
    # refuse. Quick mode exercises the code path; it does not measure anything,
    # and the numbers it prints are not interpretable.
    "allow_small_n": False,
    "law_sizes": (625, 1250, 2500, 5000, 10_000),
    "extrapolate_to": (25_000, 50_000),
}


def _gaussian_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, reflect-padded, in numpy.

    scipy.ndimage would do this in one call; it is written out because the
    kernel truncation radius is part of what is being tested. A too-short radius
    makes a large sigma behave like a smaller one, which would flatten the very
    curve this experiment reads.

    Note `mode="symmetric"`, not `"reflect"`. The two libraries use the same word
    for different things: numpy's `reflect` drops the edge sample
    (d c b | a b c d | c b a) where scipy.ndimage's `reflect` repeats it
    (d c b a | a b c d | d c b a), which is numpy's `symmetric`. Written with
    numpy's `reflect` this disagreed with `ndimage.gaussian_filter` by up to
    0.13 on [0, 1] data -- an edge artefact, but Inception sees the edges.
    """
    if sigma <= 0:
        return images.copy()
    radius = int(np.ceil(4.0 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    out = images.astype(np.float64, copy=True)
    for axis in (1, 2):
        padded = np.pad(
            out,
            [(0, 0)] + [(radius, radius) if a == axis else (0, 0) for a in (1, 2)],
            mode="symmetric",
        )
        stacked = np.stack(
            [np.take(padded, range(i, i + out.shape[axis]), axis=axis)
             for i in range(len(kernel))],
            axis=0,
        )
        out = np.tensordot(kernel, stacked, axes=(0, 0))
    return out


def _to_unit(images: np.ndarray) -> np.ndarray:
    """CIFAR luminance to [0, 1], with the scaling recorded rather than guessed.

    `inception_activations` refuses anything outside [0, 1] instead of rescaling
    silently, so the conversion happens once, here, from the *real* data's range
    and is reused for every blurred copy. Rescaling each copy by its own min/max
    would undo part of the degradation the blur is supposed to introduce.
    """
    lo, hi = float(images.min()), float(images.max())
    return np.clip((images - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _activations(images_unit, settings, label):
    stats, _, act = inception_activations(
        images_unit, batch_size=settings["batch_size"], want_probs=False,
    )
    print(f"  [{label}] {len(act)} images -> pool3 {act.shape[1]}-d "
          f"({stats.weights})")
    return act


def part_floor(settings, out_dir):
    """The noise floor, and the number every later comparison is measured against."""
    split = load_cifar10_luminance(
        settings["archive"], n_images=settings["n_images"], split="train"
    )
    unit = _to_unit(split.images)
    act = _activations(unit, settings, "real")

    out = real_vs_real_floor(
        act, n_repeats=settings["n_repeats"], seed=settings["seed"],
        allow_small_n=settings["allow_small_n"],
    )
    out["n_images"] = int(len(act))
    out["conventional_n"] = CONVENTIONAL_N
    print(f"\n  FID(real, real) = {out['floor_mean']:.4f} "
          f"+/- {out['floor_std']:.4f} over {out['n_repeats']} splits "
          f"at n={out['n_per_half']} per half")
    print("  Any model difference smaller than this is not a difference.")
    write_csv(out_dir / "floor.csv", [out])
    np.save(out_dir / "real_activations.npy", act.astype(np.float32))
    return out


def part_blur(settings, out_dir):
    """FID against blur radius. Must increase; if it does not, stop."""
    split = load_cifar10_luminance(
        settings["archive"], n_images=settings["n_images"], split="train"
    )
    unit = _to_unit(split.images)
    half = len(unit) // 2
    reference, target = unit[:half], unit[half:2 * half]

    ref_act = _activations(reference, settings, "reference")
    rows, by_sigma = [], {}
    for sigma in settings["blur_sigmas"]:
        blurred = _gaussian_blur(target, float(sigma))
        act = _activations(np.clip(blurred, 0.0, 1.0), settings, f"blur={sigma}")
        value = fid_from_samples(
            ref_act, act, allow_small_n=settings["allow_small_n"]
        )
        by_sigma[float(sigma)] = value
        rows.append({"sigma": float(sigma), "fid": value, "n_per_side": half})
        print(f"  sigma={sigma:<5} FID={value:.4f}")

    check = blur_monotonicity(by_sigma)
    print(f"\n  monotone: {check['monotone']}")
    if not check["monotone"]:
        print("  VIOLATIONS -- the metric is misconfigured. Do not score samples:")
        for v in check["violations"]:
            print(f"    sigma {v['from_sigma']} -> {v['to_sigma']}: "
                  f"FID {v['from_fid']:.4f} -> {v['to_fid']:.4f} (fell)")
    write_csv(out_dir / "blur.csv", rows)
    write_json(out_dir / "blur_check.json", check)
    return check


def part_bias(settings, out_dir):
    """FID against n on two halves of the real data: the bias, made concrete."""
    split = load_cifar10_luminance(
        settings["archive"], n_images=settings["n_images"], split="train"
    )
    unit = _to_unit(split.images)
    half = len(unit) // 2
    act_a = _activations(unit[:half], settings, "half-a")
    act_b = _activations(unit[half:2 * half], settings, "half-b")

    rows = bias_curve(
        act_a, act_b, sizes=settings["bias_sizes"],
        n_repeats=settings["n_repeats"], seed=settings["seed"],
        allow_small_n=settings["allow_small_n"],
    )
    for r in rows:
        print(f"  n={r['n']:<7} FID={r['fid_mean']:.4f} +/- {r['fid_std']:.4f}")
    if len(rows) > 1 and rows[-1]["fid_mean"] >= rows[0]["fid_mean"]:
        print("  WARNING: the curve is not falling. FID should decrease in n; "
              "a flat curve means something other than the bias dominates.")
    write_csv(out_dir / "bias.csv", rows)
    return rows


def part_biaslaw(settings, out_dir):
    """Decompose the small-n bias, and say what it extrapolates to.

    `bias` shows the floor falling as n grows. This asks *which half of the
    estimator* is doing it, because that decides whether extrapolating to a
    reporting n is legitimate or merely tidy.

    The mean term has an exactly known expectation, 2 Tr(Sigma) / n, for any
    distribution with finite covariance. The covariance term does not: it is
    estimating ~d^2/2 = 2.1M parameters from n samples, and at d = 2048 the
    n available here is the marginal regime. So the two are reported
    separately, the mean term is checked against its closed form, and the
    extrapolation is stated as what it is -- a fit to the part that dominates.
    """
    act_path = out_dir / "real_activations.npy"
    if act_path.exists():
        act = np.load(act_path).astype(np.float64)
        print(f"  reusing cached activations {act.shape}")
    else:
        split = load_cifar10_luminance(
            settings["archive"], n_images=settings["n_images"], split="train"
        )
        act = _activations(_to_unit(split.images), settings, "real")

    sigma_full = np.cov(act, rowvar=False, ddof=1)
    tr_sigma = float(np.trace(sigma_full))
    print(f"  d={act.shape[1]}  Tr(Sigma)={tr_sigma:.2f}  "
          f"n/d={len(act) / act.shape[1]:.2f}")

    rng = np.random.default_rng(settings["seed"])
    rows = []
    for n in settings["law_sizes"]:
        if 2 * n > len(act):
            continue
        parts = []
        for _ in range(settings["n_repeats"]):
            perm = rng.permutation(len(act))
            a = ActivationStats.from_activations(act[perm[:n]])
            b = ActivationStats.from_activations(act[perm[n:2 * n]])
            parts.append(frechet_decomposition(a.mu, a.sigma, b.mu, b.sigma))
        row = {
            "n": int(n),
            "fid": float(np.mean([p["fid"] for p in parts])),
            "mean_term": float(np.mean([p["mean_term"] for p in parts])),
            "cov_term": float(np.mean([p["covariance_term"] for p in parts])),
            "mean_term_predicted": predicted_mean_term(sigma_full, n),
        }
        row["mean_fraction"] = row["mean_term"] / row["fid"]
        rows.append(row)
        print(f"  n={n:<6} FID={row['fid']:8.4f}  mean={row['mean_term']:7.4f} "
              f"(predicted {row['mean_term_predicted']:7.4f})  "
              f"cov={row['cov_term']:8.4f}  mean is {100 * row['mean_fraction']:.1f}%")

    # Power law on the total. Reported with its exponent, not assumed to be 1/n:
    # a fitted exponent far from -1 would say the covariance term is not in the
    # regime the mean term is, and that an extrapolation should not be trusted.
    if len(rows) >= 3:
        ln = np.log([r["n"] for r in rows])
        lf = np.log([r["fid"] for r in rows])
        slope, intercept = np.polyfit(ln, lf, 1)
        resid = lf - (slope * ln + intercept)
        print(f"\n  fitted law: FID ~ n^({slope:.4f}), "
              f"max |log residual| {np.max(np.abs(resid)):.4f}")
        for target in settings["extrapolate_to"]:
            print(f"    extrapolated floor at n={target}: "
                  f"{np.exp(intercept + slope * np.log(target)):.4f}")
        write_json(out_dir / "biaslaw_fit.json", {
            "exponent": float(slope),
            "log_intercept": float(intercept),
            "max_abs_log_residual": float(np.max(np.abs(resid))),
            "trace_sigma": tr_sigma,
            "dimension": int(act.shape[1]),
            "extrapolated": {
                str(t): float(np.exp(intercept + slope * np.log(t)))
                for t in settings["extrapolate_to"]
            },
            "caveat": (
                "The exponent is fitted to the total. The mean term is provably "
                "1/n; the covariance term is not known to be, and it dominates. "
                "Treat any extrapolation beyond the measured range as indicative."
            ),
        })
    write_csv(out_dir / "biaslaw.csv", rows)
    return rows


PARTS = {"floor": part_floor, "blur": part_blur, "bias": part_bias,
         "biaslaw": part_biaslaw}


def main() -> None:
    parser = experiment_parser(
        "exp_30_fid_validation",
        "Validate the FID pipeline against known answers before scoring samples.",
    )
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    settings = apply_overrides(SETTINGS, args.set)
    if args.quick:
        settings.update(
            n_images=600, batch_size=32, blur_sigmas=(0.0, 1.0, 4.0),
            bias_sizes=(150, 300), n_repeats=2, allow_small_n=True,
            law_sizes=(75, 150, 300), extrapolate_to=(1000,),
        )

    out_dir = ensure_dir(args.output_dir)
    parts = select_parts(PARTS, args.only)
    for name, fn in parts.items():
        print(f"=== {name} ===")
        fn(settings, out_dir)
    write_json(out_dir / "params.json",
               {"settings": settings, "parts": list(parts),
                "provenance": provenance()})


if __name__ == "__main__":
    main()
