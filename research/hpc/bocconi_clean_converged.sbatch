#!/bin/bash
#SBATCH --job-name=clean800
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
cd $HOME/nongaussian-bp/research/nongaussian-bp
P=$HOME/nongaussian-bp/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
SIZES=(64 128 256 512 1024)
N=${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}
# Sharded by SAMPLE SIZE, not by seed: replicates are internal to this part
# (rng_for("exp06-p6", n_chains, r)) and there is no seed_offset to shard on --
# array 632877 tried and every task died on the unknown-key guard, which is
# exactly what that guard is for.
#
# shape_iters=800 clears the median 229 updates the kurtosis needs (exp_27); the
# 120 every previous run used does not, which is why those numbers were marked
# provisional in the source and never left the compendium.
echo "clean_vs_noised converged: nseq=$N reps=16 shape_iters=800"
$P experiments/exp_06_em_parameter_recovery.py \
   --only clean_vs_noised \
   --output-dir $HOME/nongaussian-bp/outputs/frozen/exp_06_clean_conv_n$N \
   --set sizes_clean="($N,)" \
   --set shape_iters_clean=800
