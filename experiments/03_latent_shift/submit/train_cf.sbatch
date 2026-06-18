#!/bin/bash
#
# Train CellFlow with configurable embedding + cell_type covariate.
#
# Usage:
#   sbatch --export=MODEL=PCA_128 runs/scripts/train/train_cf.sbatch
#   sbatch --export=MODEL=AE_128  runs/scripts/train/train_cf.sbatch
#   sbatch --export=MODEL=AE_512  runs/scripts/train/train_cf.sbatch
#   sbatch --export=MODEL=AE_512,NUM_ITERS=1000000 runs/scripts/train/train_cf.sbatch
#

#SBATCH --job-name=cf_train
#SBATCH -o logs/slurm/%x_%j_out.txt
#SBATCH -e logs/slurm/%x_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --time=7:00:00
#SBATCH --nice=0

# ── environment ─────────────────────────────────────────────────────────
EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/.bashrc"
conda deactivate
conda activate pancellflow
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── paths ───────────────────────────────────────────────────────────────
REPO=${EXPDIR}

MODEL=${MODEL:-AE_2048}
CONFIG=${CONFIG:-repro}
NUM_ITERS=${NUM_ITERS:-500000}
BATCH_SIZE=${BATCH_SIZE:-1024}
VALID_FREQ=${VALID_FREQ:-50000}
OUT_DIR=${OUT_DIR:-}

mkdir -p "${REPO}/logs/slurm"

echo "=== CellFlow training: ${MODEL} (config=${CONFIG}) ==="
echo "Iterations: ${NUM_ITERS}, Batch: ${BATCH_SIZE}, Valid freq: ${VALID_FREQ}"

# ── run ─────────────────────────────────────────────────────────────────
CMD="python -u ${REPO}/codes/train_cf.py \
    --model ${MODEL} \
    --config ${CONFIG} \
    --num_iters ${NUM_ITERS} \
    --batch_size ${BATCH_SIZE} \
    --valid_freq ${VALID_FREQ}"
[ -n "${OUT_DIR}" ] && CMD="${CMD} --out_dir ${OUT_DIR}"
eval ${CMD}

echo "=== Done ==="
