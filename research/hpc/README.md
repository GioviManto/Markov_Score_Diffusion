# Running the EM experiments on a cluster

The Layer-5 experiments are CPU-bound, pure numpy, and embarrassingly parallel
at the level of *parts*: each part of `exp_06` / `exp_07` reads nothing the
others write and emits its own CSV/PNG into the shared output directory. So a
job array with one task per part needs no merge step — the outputs simply
appear next to each other.

This directory contains SLURM array scripts. **They are templates**: the lines
marked `# SITE` need your cluster's account, partition, and module names, since
those cannot be guessed. If your cluster runs PBS/Torque or LSF instead, the
body of each script (the `srun`/`python` line) transfers unchanged — only the
directive block and the array-index variable differ.

## Quick orientation

```bash
python experiments/exp_06_em_parameter_recovery.py --list-parts
python experiments/exp_07_em_vs_score_network.py  --list-parts
```

Run one part:

```bash
python experiments/exp_06_em_parameter_recovery.py --only rate
```

Scale a part up (any key printed in `params_*.json` can be overridden):

```bash
python experiments/exp_06_em_parameter_recovery.py --only rate \
    --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,4096,16384)'
```

Unknown keys are rejected rather than ignored, so a typo in a job script fails
immediately instead of quietly running the default configuration and looking
like a finished experiment.

## Threading

Every part is dominated by `(M, M) @ (M, B)` matmuls, so BLAS threading does
most of the work within a task. Give each array task several cores and set the
thread count to match — the scripts do this from `SLURM_CPUS_PER_TASK`. Do not
leave `OMP_NUM_THREADS` unset: the default is one thread per core on the whole
node, and concurrent array tasks will then oversubscribe and slow each other
down.

## What is worth the extra compute

The single-node runs committed in `outputs/` were sized to finish in about an
hour each. The parts that would genuinely benefit from a cluster, in order:

1. **`exp_07 sample_efficiency`** — the headline curve has one replicate per
   `N`, which is why the EM curve wobbles. Repeating it over seeds would turn
   it into a curve with error bars, which is what a paper needs. Extend `sizes`
   upward too: the network was still improving at `N = 2048`, and the honest
   version of "EM-BP needs ≥64× less data" requires following the network until
   it either catches up or visibly plateaus.
2. **`exp_06 rate`** — the `N^{-1/2}` claim rests on 4 replicates. It should
   have 32+, and a decade more range in `N`.
3. **`exp_06 price_of_noising`** — 10 replicates make the sd/CRLB ratios noisy
   to about ±25%. More replicates would let the efficiency claim be stated
   sharply rather than as "within ±30%".
4. **`exp_06 quantization`** — the `M = 1601` grid is the expensive one, and
   the lattice effect would be better characterized on a denser sweep of
   `rho_true` values that are *not* low-denominator rationals.
5. **`exp_07 capacity`** — larger architectures and longer training, to push the
   "the network is not merely undertrained" argument as far as it will go.

### Layer 6 (exp_13, exp_14)

6. **`exp_14 collapse`** — the one that most needs a cluster, because the point
   is to straddle the entropic wall and the wall is exponential in the chain
   length. At `rho = 0.85` it sits at ~89 chains for `n = 8` and ~1e9 for
   `n = 33`, so only the short chains can be taken past it at all. A sweep that
   crosses the wall for `n = 8` and `n = 12`, and demonstrably cannot for
   `n = 24` and `n = 32`, is the whole argument in one figure.
7. **`exp_13 levels` / `ordering`** — these are gated by tree EM, which is
   several times slower to converge than chain EM (internal nodes are never
   observed). The committed runs use `em_iters = 150` at depth 4; depth 5-6
   with `em_iters = 400` is what the cluster is for. **Do not economize here**:
   an under-budgeted EM produces a plausible wrong answer rather than an
   obvious failure.
8. **`exp_13 cascade`** — cheap per path but the finest level of the ladder is
   integrator-limited (measured 0.125 against a predicted 0.087 at `t_min =
   0.02`). More steps and a smaller `t_min` should close that; if it does not,
   that is itself worth knowing.

## Files

    slurm_exp06.sbatch   5-task array, one per part of exp_06
    slurm_exp07.sbatch   4-task array, one per part of exp_07
    slurm_layer6.sbatch  8-task array: 5 parts of exp_13, 3 of exp_14
    slurm_replicates.sbatch
                         seed-replicated sample-efficiency sweep: one task per
                         seed, each writing to its own subdirectory, plus a
                         merge helper for producing error bars

---

## GPU execution (added August 2026)

Grid BP runs on GPU through `src/backend.py`. There is **one** implementation of the
recursion, parameterised by its array module -- not a CPU version and a GPU version. A
separately written device kernel would put the exactness guarantees (9.2e-15 against the
closed-form Gaussian score, 1.0e-14 against brute-force enumeration) at risk in the least
visible way possible.

`tests/test_backend_parity.py` gates every GPU number and `hpc/bocconi_gpu.sbatch` runs it
before the sweep, exiting rather than proceeding if it fails. Verified on an NVIDIA H200:

| check | result |
|---|---|
| CPU/GPU posterior means, 2 families x 3 noise levels | 1.8e-16 to 8.0e-16 relative |
| CPU/GPU posterior variances | 3.5e-16 to 9.3e-15 |
| GPU vs closed-form Gaussian score | 4.2e-16 to 1.3e-13 |
| dtype preserved | float64 |
| result independent of batch width | 12/12 tests pass |

### The version pin is load-bearing

**`cupy-cuda12x<14`.** CuPy 14.x JIT-compiles its cub segmented-reduction kernels against
headers this cluster's toolkit cannot build -- 42 errors inside `cuda_fp4.hpp`. The failure
mode is what makes it dangerous rather than merely annoying: cuBLAS matmuls have no JIT step
and keep working, so the device looks healthy until the first `.max(axis=...)` inside the
recursion. `cupy 13.6.0` compiles every operation the recursion needs; an operation-by-
operation probe is in `diag/probe.py`.

Environment is pinned by `gpu_env.sh`: `CUDA_PATH=/software/cuda/12.4` (the module's
`lib64`, not `lib`), *not* the system default at `/usr/local/cuda-13.3`.

Also note `/tmp` is **not** shared between login and compute nodes -- scripts must live under
`$HOME`.
