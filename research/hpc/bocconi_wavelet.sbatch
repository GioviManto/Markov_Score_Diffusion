#!/bin/bash
#SBATCH --job-name=wavelet
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_23/25 (real images) and exp_26 (video) -- the wavelet-tree extension.
#
# TIME LIMIT. `medium_cpu` caps at 6:10:00 (sinfo), so the 07:45:00 this script
# asked for in its previous version would have been rejected at submit rather
# than run and truncated. 05:45 leaves margin under the cap. Anything longer
# belongs on `compute` (3-00:10:00, 3 nodes) -- see MODE=video below.
#
# COST, measured rather than estimated. One BP pass is ~30 ms per
# image-orientation on a uniform M=241 grid; per-depth grids resolving to
# t = 0.05 cost about the same in total while reaching 18x further down in t
# (they spend points where the likelihood is narrow, which is where the nodes
# are few). Measured on this laptop:
#
#   exp_25 fit: 600 images, 12 iters, 3 orientations, 2 noise levels -> ~9 min
#               so n_train=2000, n_iters=20 -> ~50 min per family
#   exp_26 fit: 300 sequences x 6 frames, 8 iters                    -> ~20 min
#               (the caterpillar traverses each frame's tree twice)
#   exp_26 cutcurve: ~25 min per cut depth at those settings
#
# Cost is linear in n_train and n_iters and quadratic in grid size; the grid
# sizes follow from t_resolve, so *lowering t_resolve raises cost sharply*.
# Re-run the arithmetic before raising anything.
#
# LOG BUFFERING. Every python call uses `-u`. Without it CPython block-buffers
# stdout to a file, so a multi-hour job writes nothing until it exits and a hung
# job is indistinguishable from a quiet one.
#
# USAGE
#
#   Gate first (tests + every experiment at --quick, ~20 min):
#     SMOKE=$(sbatch --parsable --partition=short_cpu --time=01:00:00 \
#             --export=ALL,MODE=smoke bocconi_wavelet.sbatch)
#
#   Images -- the gating measurement on 10k CIFAR images:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=stats bocconi_wavelet.sbatch
#
#   Images -- generation, one array task per kernel family:
#     sbatch --dependency=afterok:$SMOKE --array=0-2 --export=ALL,MODE=generate \
#            bocconi_wavelet.sbatch
#
#   Video -- the space/time trade-off, one array task per cut depth:
#     sbatch --dependency=afterok:$SMOKE --array=0-4 --export=ALL,MODE=cutcurve \
#            bocconi_wavelet.sbatch
#
#   Video -- caterpillar vs frozen control, coherence, structure, strips:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=video bocconi_wavelet.sbatch
#
# COMBINED MODES, for when the queue is full. The QOS caps submissions at 30 per
# user and array tasks count individually, so an 11-job plan does not fit beside
# other work. `images` and `videoall` do the same runs sequentially in one job
# each, which costs wall-clock but only one slot. They need the `compute`
# partition (3-00:10:00); medium_cpu at 6:10 is not enough for either.
#
#     sbatch --partition=compute --time=12:00:00 --export=ALL,MODE=images   bocconi_wavelet.sbatch
#     sbatch --partition=compute --time=23:00:00 --export=ALL,MODE=videoall bocconi_wavelet.sbatch
#
#   exp_24 with the scale_mixture kernel added -- calibrate on THIS cluster before
#   guessing the full-size walltime, then submit the full-size run on compute:
#     sbatch --export=ALL,MODE=imagefit_scale_calibrate bocconi_wavelet.sbatch
#     sbatch --partition=compute --time=3-00:00:00 \
#            --export=ALL,MODE=imagefit_scale bocconi_wavelet.sbatch
#
#   GPU. The quadtree recursion runs on a device as of 19 Aug 2026. Gate first --
#   it is 15 minutes and it is the only thing standing between a port bug and a
#   plausible-looking wrong number -- then the full-size fit:
#     sbatch --partition=debug_gpuh200 --qos=debug --gres=gpu:1 --time=00:15:00 \
#            --propagate=NONE --export=ALL,MODE=gpu_gate bocconi_wavelet.sbatch
#     sbatch --partition=medium_gpuh200 --gres=gpu:1 --time=06:00:00 \
#            --propagate=NONE --export=ALL,MODE=imagefit_scale_gpu bocconi_wavelet.sbatch
#
# --propagate=NONE ON EVERY SUBMISSION. The login node sets `ulimit -t 600` and
# Slurm propagates the submitter's rlimits, so a 6-hour job without it is killed
# by SIGXCPU after ten minutes of CPU time -- with nothing in the log, and
# `sacct` showing TotalCPU pinned at exactly 10:00. An array died to this on
# 18 Aug 2026.
#
# The image array splits by *family* and the video array by *cut depth*, because
# in both cases the tasks share nothing: each fits its own model and writes its
# own file, so there is no merge step. Splitting by experiment *part* instead
# would refit the same models once per part.

set -euo pipefail

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-stats}

# GPU gate, lifted from bocconi_frozen.sbatch's MODE=gpu and kept identical on
# purpose. The wavelet recursion became device-parameterised on 19 Aug 2026;
# before that `src/backend` was wired into bp_grid and denoiser only, so
# BP_DEVICE=gpu was a silent no-op for every experiment in this file.
#
# The gate checks the DEVICE, not pytest's exit code. Every test in the parity
# suite skips when cupy cannot see a card, an all-skipped run exits 0, and the
# job would then do hours of CPU work wearing a GPU job's name -- which is what
# job 631477 did on an H200. So: prove the device reduces correctly, then run
# parity, then refuse a run in which anything skipped.
gpu_gate() {
    [ -f "$ROOT/gpu_env.sh" ] && source "$ROOT/gpu_env.sh"
    export BP_DEVICE=gpu

    "$PYTHON" - <<'PY' || { echo "FATAL: BP_DEVICE=gpu but no usable CUDA device" >&2; exit 1; }
import sys
from src.backend import gpu_available
if not gpu_available():
    sys.exit(1)
# A reduction, not just a matmul: cupy-cuda12x JIT-compiles reduction kernels and
# that compile fails against the system-default CUDA 13.3 headers while cuBLAS
# keeps working, so a matmul-only probe passes on a node where every .sum() in
# the recursion would die mid-sweep.
import cupy as cp
a = cp.arange(1000, dtype=cp.float64)
assert abs(float(a.sum()) - 499500.0) < 1e-6, "cupy reduction returned the wrong value"
PY

    "$PYTHON" -m pytest tests/test_backend_parity.py -q -rs --no-header \
        -o cache_dir=/tmp/pytest_cache | tee /tmp/parity_wavelet.log
    if grep -qE '[0-9]+ skipped' /tmp/parity_wavelet.log; then
        echo "FATAL: parity tests skipped on a node with a working device" >&2
        exit 1
    fi
    echo "GPU GATE PASSED"
}

THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

# The CIFAR archive is gitignored, so it will not have arrived with the code.
ARCHIVE=$ROOT/research/nongaussian-bp/data/cifar-10-python.tar.gz
if [ ! -s "$ARCHIVE" ]; then
    echo "missing $ARCHIVE" >&2
    echo "fetch it once on the login node:" >&2
    echo "  mkdir -p $(dirname "$ARCHIVE")" >&2
    echo "  curl -L -o $ARCHIVE https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" >&2
    echo "  md5sum $ARCHIVE   # expect c58f30108f718f92721af3b95e74349a -- REFUSE the file if it differs" >&2
    exit 2
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE  threads=$THREADS"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    "$PYTHON" -u -m pytest tests/ -q -m "not slow"
    for e in exp_23_wavelet_statistics exp_24_wavelet_fit \
             exp_25_wavelet_generation exp_26_video; do
        echo "--- $e --quick ---"
        "$PYTHON" -u "experiments/$e.py" --quick \
            --output-dir "$ROOT/outputs/smoke/$e"
    done
    ;;

  stats)
    # The gate: if CIFAR subband coefficients came back approximately Gaussian
    # the whole premise would not apply, so this runs first and at full size.
    "$PYTHON" -u experiments/exp_23_wavelet_statistics.py \
        --output-dir "$ROOT/outputs/exp_23_wavelet_statistics" \
        --set n_images=10000 --set n_boot=400
    ;;

  generate)
    FAMILIES=(gaussian mixture scale_mixture)
    F=${FAMILIES[$TASK]}
    OUT=$ROOT/outputs/exp_25_wavelet_generation/$F
    echo "family=$F -> $OUT"
    "$PYTHON" -u experiments/exp_25_wavelet_generation.py \
        --output-dir "$OUT" \
        --set families="('$F',)" \
        --set n_train=2000 --set n_test=5000 --set n_iters=20 \
        --set t_resolve=0.05 --set t_train="(0.3,0.6,1.0)" \
        --set t_likelihood=0.05 \
        --set n_generate=10000 --set n_reverse=128 --set reverse_steps=150
    ;;

  cutcurve)
    # One cut depth per task. Each is an independent fit of a *different exact
    # model* -- the space/time edge trade of video_bp.cut_caterpillar_bp -- and
    # writes cutcurve_cut<N>.csv, so the five concatenate with no merge logic.
    OUT=$ROOT/outputs/exp_26_video
    echo "cut_depth=$TASK -> $OUT"
    "$PYTHON" -u experiments/exp_26_video.py --only cutcurve \
        --output-dir "$OUT" \
        --set cuts="($TASK,)" \
        --set n_train=1000 --set n_test=1000 --set n_iters=12 \
        --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    ;;

  video)
    # Caterpillar against the frozen-time control, plus the coherence
    # decomposition, the loop-free budget, and the frame strips.
    "$PYTHON" -u experiments/exp_26_video.py \
        --only fit,coherence,structure,strip \
        --output-dir "$ROOT/outputs/exp_26_video" \
        --set n_train=1000 --set n_test=1000 --set n_iters=12 \
        --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    ;;

  imagefit)
    # exp_24 at full size. It is not part of `images` and was not part of any
    # other mode either, so on 2026-08-12 it was the single experiment in the
    # package with no outputs/ directory at all -- only the few seconds of it the
    # smoke gate runs. That is the wrong thing to have missing: exp_24 is the
    # Gaussian tree baseline, the second-order closure every non-Gaussian result
    # on images is quoted against, and its own docstring calls it "worth having
    # on its own".
    #
    # Defaults (n_train=2000, n_test=500, n_iters=25, families gaussian+mixture)
    # rather than anything larger, because the point is the baseline. This mode
    # writes gaussian+mixture only, unchanged -- see `imagefit_scale` below for
    # the scale_mixture extension, kept as a separate output so this baseline
    # stays reproducible on its own.
    # IMAGEFIT_OUT lets a second, longer-walltime attempt run alongside the first
    # without the two writing to one directory. exp_24 accumulates both families
    # before writing its fit CSV, so a walltime kill loses the completed family
    # too -- there is no partial credit here, which is why a backup attempt is
    # worth a slot. Measured 2026-08-12: the gaussian fit alone had not finished
    # after 2h at 357% CPU on medium_cpu's 6h cap.
    "$PYTHON" -u experiments/exp_24_wavelet_fit.py \
        --output-dir "$ROOT/outputs/${IMAGEFIT_OUT:-exp_24_wavelet_fit}"
    ;;

  imagefit_scale_calibrate)
    # `_factory` in exp_24 accepted only gaussian+mixture; `src/scale_kernel.py`'s
    # ScaleMixtureKernel (added for exp_25) is now wired in too, so the second-order
    # closure can finally be compared against the family built to capture cross-scale
    # magnitude dependence -- the gap EXECUTIVE_SUMMARY.md's own limitations section names.
    #
    # `imagefit`'s gaussian family alone had not finished after 2h on medium_cpu's 6h cap
    # at full size (n_train=2000, n_iters=25), and exp_24 fits every family in `families`
    # sequentially in one process before writing anything -- no partial credit on a
    # walltime kill. Do not submit the full three-family run at full size blind: this mode
    # measures actual per-iteration, per-family cost on THIS cluster at a small size first
    # (n_train=200, n_iters=5), so `imagefit_scale`'s walltime request is sized from a
    # measurement instead of a guess.
    "$PYTHON" -u experiments/exp_24_wavelet_fit.py \
        --output-dir "$ROOT/outputs/exp_24_scale_calibrate" \
        --set families="('gaussian','mixture','scale_mixture')" \
        --set n_train=200 --set n_test=100 --set n_iters=5
    ;;

  imagefit_scale)
    # The full-size, three-family rerun -- submit only after inspecting
    # imagefit_scale_calibrate's elapsed time and confirming this walltime request has
    # margin. Same n_train/n_test/n_iters as `imagefit` (2000/500/25) so the scale_mixture
    # result is comparable to the committed gaussian+mixture baseline, not a smaller,
    # different measurement. Writes to its own directory: exp_24_wavelet_fit (the
    # gaussian+mixture baseline) is not touched.
    "$PYTHON" -u experiments/exp_24_wavelet_fit.py \
        --output-dir "$ROOT/outputs/${IMAGEFIT_OUT:-exp_24_wavelet_fit_scale_mixture}" \
        --set families="('gaussian','mixture','scale_mixture')"
    ;;

  gpu_gate)
    # Prove the ported quadtree recursion agrees across devices on THIS card, and
    # measure what the port actually buys, before any long GPU run is submitted.
    # Cheap enough for debug_gpuh200 (15 min).
    gpu_gate
    echo
    echo "=== CPU vs GPU, one exp_24-shaped E-step ==="
    "$PYTHON" -u - <<'PY'
import time
import numpy as np
from src.backend import get_xp, device_name
from src.image_data import load_cifar10_luminance
from src.wavelet import images_to_tree
from src.wavelet_model import SubbandScales, per_depth_grid_sizes
from src.bp_grid import make_grid
from src.wavelet_bp import wavelet_tree_bp
from src.kernels import GaussianAR1Kernel
from src.noising import alpha_delta

# The real shape: CIFAR luminance, 5 levels, the per-depth mesh exp_24 resolves
# to t=0.05. Timing a toy tree would say nothing about the batch/grid regime the
# fit actually runs in.
imgs = load_cifar10_luminance(
    "data/cifar-10-python.tar.gz", n_images=256, split="train"
).images
qt, nodes, _ = images_to_tree(imgs, 5)
scales = SubbandScales.fit(qt, nodes)
std = scales.standardise(qt, nodes)
sizes = per_depth_grid_sizes(scales.scales, 0.05, 8.0)
made = [make_grid(8.0, m) for m in sizes]
grids, wts = [g for g, _ in made], [w for _, w in made]
kern = [GaussianAR1Kernel(rho=0.4, q=0.84) for _ in range(qt.depth)]
log_k = [k.log_transition_matrix(grids[d], grids[d + 1]) for d, k in enumerate(kern)]
log_root = -0.5 * grids[0] ** 2 - 0.5 * float(np.log(2 * np.pi))
alpha, delta = alpha_delta(0.3)
deltas = scales.delta_by_depth(0, delta)
print(f"grid sizes by depth: {sizes}   batch={len(imgs)}  nodes={qt.depth}")

res = {}
for label, xp in (("cpu", np), ("gpu", get_xp("gpu"))):
    t0 = time.perf_counter()
    r = wavelet_tree_bp(grids, wts, log_k, log_root, std[:, 0, :], alpha,
                        deltas, qt.branching, qt.depth, want_stats=True, chunk=64,
                        xp=xp)
    secs = time.perf_counter() - t0
    res[label] = r
    print(f"  {label:3s} ({device_name(xp)}): {secs:.2f}s   log_ev {r.log_evidence:.9e}")

c, g = res["cpu"], res["gpu"]
rel_m = np.linalg.norm(g.posterior_mean - c.posterior_mean) / np.linalg.norm(c.posterior_mean)
rel_e = abs(g.log_evidence - c.log_evidence) / abs(c.log_evidence)
print(f"  agreement: rel mean {rel_m:.3e}  rel log-evidence {rel_e:.3e}")
assert rel_m < 1e-12 and rel_e < 1e-12, "device disagreement on the real problem shape"
print("AGREEMENT OK")
PY
    ;;

  imagefit_scale_gpu)
    # The full-size three-family fit on a GPU. Same settings as `imagefit_scale`
    # so the two are directly comparable -- the device is the only difference,
    # and the parity gate above is what licenses saying that.
    #
    # This mode exists because the CPU arithmetic does not fit the queue: measured
    # on this cluster at n_train=200/n_iters=5 (job 631488), gaussian took 176s and
    # mixture 227s, which scale by 50x to ~2.4h and ~3.2h. Three families is well
    # past medium_cpu's 6h cap before scale_mixture -- the most expensive of them --
    # is counted at all.
    gpu_gate
    # EXTRA_SET passes further --set overrides through, unquoted so it word-splits
    # into separate arguments. Without it this mode silently ignored anything
    # appended at submit time -- trailing sbatch arguments go to the script, which
    # never read "$@" -- so a run asking for n_iters=120 would quietly produce
    # another n_iters=25 fit in a directory named as though it were something else.
    #   --export=ALL,MODE=imagefit_scale_gpu,IMAGEFIT_OUT=...,EXTRA_SET="--set n_iters=120"
    "$PYTHON" -u experiments/exp_24_wavelet_fit.py \
        --output-dir "$ROOT/outputs/${IMAGEFIT_OUT:-exp_24_wavelet_fit_scale_mixture_gpu}" \
        --set families="('gaussian','mixture','scale_mixture')" \
        ${EXTRA_SET:-}
    ;;

  fid_validate)
    # exp_29: gate the FID pipeline against known answers before it scores any
    # generated sample. Runs no model.
    #
    # A DIFFERENT INTERPRETER. `.venv-metrics` carries torch 2.6.0+cu124; the
    # package venv carries cupy 13.6 against the CUDA 12.4 module, and that pin is
    # load-bearing for every exact-BP number the project has (cupy 14.x passes
    # matmuls and fails reductions). Two CUDA stacks in one environment is a risk
    # with no upside here: nothing in exp_29 imports src.wavelet*, so the split
    # costs a duplicated numpy and buys immunity.
    METRICS_PY=$ROOT/.venv-metrics/bin/python
    if [ ! -x "$METRICS_PY" ]; then
        echo "missing $METRICS_PY -- create it on the login node:" >&2
        echo "  python3 -m venv $ROOT/.venv-metrics" >&2
        echo "  $ROOT/.venv-metrics/bin/pip install torch torchvision \\" >&2
        echo "      --index-url https://download.pytorch.org/whl/cu124" >&2
        echo "  $ROOT/.venv-metrics/bin/pip install numpy scipy" >&2
        exit 2
    fi
    [ -f "$ROOT/gpu_env.sh" ] && source "$ROOT/gpu_env.sh"
    "$METRICS_PY" -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    "$METRICS_PY" -u experiments/exp_30_fid_validation.py \
        --output-dir "$ROOT/outputs/exp_30_fid_validation" \
        ${FID_ARGS:-}
    ;;

  images)
    # Everything on real CIFAR in one slot: the gate, then each kernel family.
    # Sequential because the QOS cap makes an array unaffordable, not because
    # the tasks depend on each other -- they do not.
    echo "=== stats ==="
    "$PYTHON" -u experiments/exp_23_wavelet_statistics.py \
        --output-dir "$ROOT/outputs/exp_23_wavelet_statistics" \
        --set n_images=10000 --set n_boot=400
    for F in gaussian mixture scale_mixture; do
        echo "=== generate: $F ==="
        "$PYTHON" -u experiments/exp_25_wavelet_generation.py \
            --output-dir "$ROOT/outputs/exp_25_wavelet_generation/$F" \
            --set families="('$F',)" \
            --set n_train=2000 --set n_test=5000 --set n_iters=20 \
            --set t_resolve=0.05 --set t_train="(0.3,0.6,1.0)" \
            --set t_likelihood=0.05 \
            --set n_generate=10000 --set n_reverse=128 --set reverse_steps=150
    done
    ;;

  videoall)
    # The whole video story in one slot: the space/time trade at every cut
    # depth, then the caterpillar against its frozen control with the coherence
    # decomposition, the loop-free budget and the frame strips.
    OUT=$ROOT/outputs/exp_26_video
    for C in 0 1 2 3 4; do
        echo "=== cutcurve: cut_depth=$C ==="
        "$PYTHON" -u experiments/exp_26_video.py --only cutcurve \
            --output-dir "$OUT" \
            --set cuts="($C,)" \
            --set n_train=1000 --set n_test=1000 --set n_iters=12 \
            --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    done
    echo "=== caterpillar vs frozen, coherence, structure, strips ==="
    "$PYTHON" -u experiments/exp_26_video.py \
        --only fit,coherence,structure,strip \
        --output-dir "$OUT" \
        --set n_train=1000 --set n_test=1000 --set n_iters=12 \
        --set n_frames=6 --set t_resolve=0.05 --set n_generate=4000
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, stats, generate, cutcurve," >&2
    echo "                    video, images, videoall, imagefit," >&2
    echo "                    imagefit_scale_calibrate, imagefit_scale, fid_validate," >&2
    echo "                    gpu_gate or imagefit_scale_gpu)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
