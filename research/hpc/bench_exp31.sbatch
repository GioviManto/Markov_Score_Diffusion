#!/bin/bash
#SBATCH --job-name=e31bench
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#
# Measure, do not guess. exp_31 has two cost centres with opposite device
# behaviour and the job sizing depends on which dominates:
#
#   EM + reference bundles   go through src.backend, so an H200 helps
#   seq_nets training        is pure numpy and will NOT use the GPU
#
# Sizing a three-day job from laptop timings would be guessing at both. This
# prints per-iteration costs for each arm on the actual node so the screening
# and confirmatory jobs can be given real wall-clock limits.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC to the deploy dir}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "host $(hostname)  device ${BP_DEVICE:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DEV in cpu gpu; do
  echo "=============== BP_DEVICE=$DEV ==============="
  BP_DEVICE=$DEV $PY - <<'PYEOF'
import os, sys, time
sys.path.insert(0, "experiments")
import numpy as np
from frozen_config import FROZEN
from src.backend import get_xp, device_name
from src.bp_grid import make_grid
from src.em import fit_em
from src.kernels import MixtureInnovationKernel
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.seq_nets import BiMessagePassing, DilatedConv1d, train_sequence_net

print("  xp =", device_name(get_xp()))
grid, w = make_grid(8.0, 401)
prior = LaplaceAR1(0.85)
T = FROZEN.t_grid
rng = np.random.default_rng(0)

sys.path.insert(0, ".")
from exp_31_structured_baseline import _bundle
t0 = time.perf_counter(); _bundle(prior, grid, w, "bench", 0, 1024, T)
print(f"  reference bundle n=1024 : {time.perf_counter()-t0:7.1f} s")
t0 = time.perf_counter(); _bundle(prior, grid, w, "bench", 1, 2048, T)
print(f"  reference bundle n=2048 : {time.perf_counter()-t0:7.1f} s")

for n in (512, 2048):
    A = np.stack([prior.sample(rng, 32) for _ in range(n)])
    idx = np.array_split(rng.permutation(n), len(T))
    groups = []
    for t, i in zip(T, idx):
        al, de = alpha_delta(t); s = A[i]
        groups.append((al * s + np.sqrt(de) * rng.standard_normal(s.shape), al, de))
    t0 = time.perf_counter()
    fit_em(MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng),
           grid, w, groups, n_iters=8, tol=0.0)
    dt = (time.perf_counter() - t0) / 8
    print(f"  EM n={n:<5}            : {dt:7.2f} s/iter -> 1200 = {dt*1200/3600:5.2f} h")

A = np.cumsum(rng.standard_normal((512, 32)) * 0.5, axis=1)
for label, net in (("conv c=64 ", DilatedConv1d.init(64, (1, 2, 4, 8), rng)),
                   ("conv c=128", DilatedConv1d.init(128, (1, 2, 4, 8), rng)),
                   ("bimp h=64 ", BiMessagePassing.init(64, rng)),
                   ("bimp h=128", BiMessagePassing.init(128, rng))):
    t0 = time.perf_counter()
    train_sequence_net(net, A, T, rng, checkpoints=(150,), batch_size=64, lr=1e-3)
    dt = (time.perf_counter() - t0) / 150
    print(f"  {label}             : {dt*1000:7.2f} ms/step -> 64k = {dt*64000/60:5.1f} min")
PYEOF
done
echo "BENCH DONE"
