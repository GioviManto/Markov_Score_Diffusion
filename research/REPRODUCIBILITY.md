# Reproducibility

Everything below was run on this branch. Paths are relative to
`research/nongaussian-bp/` unless stated otherwise.

## Environment

The package is pure NumPy/SciPy on CPU; CuPy is optional and only accelerates the batched
grid recursion.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Versions used for the committed outputs: Python 3.12, NumPy 2.4, SciPy 1.17, Matplotlib 3.10 —
except `outputs/exp_08_gradient_vs_exact_mstep/`, regenerated on 10 August 2026 under Python
3.12.5 / NumPy 2.5.2 / SciPy 1.18.0 after the trace-alignment fix. Its `params.json` records
the newer versions, so the difference is visible rather than implicit.
The HPC runs used Python 3.11 / NumPy 2.4 on Linux; provenance (Python, NumPy, platform,
hostname, SLURM ids, thread counts) is recorded in the `params*.json` beside every output
directory, so a number can always be traced to the machine that produced it.

GPU is optional. It requires `cupy<14` — CuPy 14's cub JIT fails against this cluster's headers —
with `CUDA_PATH` pointing at CUDA 12.4:

```bash
pip install "cupy-cuda12x<14"
export CUDA_PATH=/software/cuda/12.4
```

`src/backend.py` probes the GPU with a **matmul**, not an allocation: allocation succeeds on a
node with no working cuBLAS, so probing it would report a GPU that then fails mid-sweep.

## Quick check (about two minutes)

```bash
OMP_NUM_THREADS=4 python3 -m pytest tests/ -q
```

Expected: `309 passed, 12 skipped` (measured 2026-08-12, 11:42 on an M-series laptop; Python
3.13, NumPy 2.5.1, SciPy 1.18.0). This file previously said `236 passed, 12 skipped` in ~2 min,
which was correct before the wavelet and video suites merged in and stale for some days
afterwards — the count is the first thing to drift after a merge, so treat a mismatch as a
prompt to re-measure rather than as a failure. The skips are the CuPy parity tests, skipped when
no GPU is present; they pass 12/12 on the cluster's H200 nodes. These tests are the evidence behind several numbers quoted in the note —
the brute-force enumeration check, the message-normalisation convention, the Fisher-identity
gradient check — so a failure here invalidates part of the write-up, not just the code.

## Regenerating the revision diagnostics

These three produce the outputs the revision added. Total about 30 minutes on one core.

```bash
OMP_NUM_THREADS=4 python3 experiments/exp_18_revision_diagnostics.py --parts boundary
```
```bash
OMP_NUM_THREADS=4 python3 experiments/exp_18_revision_diagnostics.py --parts emtrace
```
```bash
OMP_NUM_THREADS=4 python3 experiments/exp_18_revision_diagnostics.py --parts density
```

Writing to `outputs/exp_18/{boundary,em_trace,innovation_density,innovation_summary}.csv`.

## Regenerating every figure

```bash
python3 tools/make_figures.py
```

Reads only committed CSVs under `outputs/` and writes PDF and PNG into `overleaf/shared/figures/`. It
exits non-zero and names the missing file if any input is absent, so a figure can never be
silently stale with respect to its data.

## Compiling the documents

Both compile with `tectonic` (used here) or a full TeX Live via `latexmk`.

```bash
cd ../../overleaf/paper && tectonic -k main.tex
```
```bash
cd ../../overleaf/compendium && tectonic -k main.tex
```

With TeX Live instead:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The advisor documents are rendered from their Markdown sources:

```bash
python3 tools/md2tex.py EXECUTIVE_SUMMARY.md ANSWERS_AND_QUESTIONS_FOR_ADVISORS.md
```

## Report-quality experiments (local, hours)

Each script takes `--parts` to run a subset and writes to `outputs/<name>/`.

```bash
python3 experiments/exp_01_grid_validation.py
```
```bash
python3 experiments/exp_02_laplace_gaussian_message_error.py
```
```bash
python3 experiments/exp_06_em_parameter_recovery.py
```
```bash
python3 experiments/exp_07_em_vs_score_network.py --parts sample_efficiency,capacity,cost
```

## HPC (Bocconi)

Host `$BP_CLUSTER_HOST`, user `$BP_CLUSTER_USER` (both from the
environment; see `hpc/`), project root
`~/nongaussian-bp`, dedicated venv. QOS caps are 30 submitted and 10 running per user, so
array sizes above 30 are rejected outright rather than queued.

Calibration must run and be inspected **before** the main generation sweep: the reference arm
does not reproduce the true innovation kurtosis at small step counts, so a comparison between
arms at an unvalidated step count measures the integrator rather than the score.

```bash
sbatch --array=0-5 --export=ALL,MODE=calibrate hpc/bocconi_exp16.sbatch
```
```bash
sbatch --array=0-15 --export=ALL,MODE=generate,NSTEPS=400 hpc/bocconi_exp16.sbatch
```
```bash
sbatch --array=0-4 --export=ALL,MODE=cpoint hpc/bocconi_cpoint.sbatch
```

The GPU script runs the CPU/GPU parity suite first and exits non-zero if it fails, so a sweep
cannot start on a node whose numerics disagree with the reference:

```bash
sbatch hpc/bocconi_gpu.sbatch
```

Pulling results back:

```bash
rsync -avz "$BP_CLUSTER_USER@$BP_CLUSTER_HOST":'~/nongaussian-bp/outputs/exp_16/' outputs/exp_16/
```

## Which output supports which claim

`overleaf/paper/appendix.tex`, Appendix J, is the claim-to-script-to-output table, and
`REVISION_AUDIT.md` (archived; see `CLAIMS_TO_UPDATE.md` for the current claim ledger) records
the audit that produced it, including the one number found not to be traceable to any committed
run and what replaced it.
