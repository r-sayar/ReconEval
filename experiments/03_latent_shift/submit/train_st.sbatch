#!/bin/bash
#
# State Transition training — one job per embedding model.
# Supports two decoder modes:
#   frozen (default): pre-trained frozen adapter (MLP/PCA/AE/nlscVI)
#   fresh:            jointly-trained LatentToGeneDecoder, paper's ST+SE approach
#
# Usage (single model):
#   sbatch --export=MODEL=AE_2048 runs/scripts/train/st_perturbation.sbatch
#   sbatch --export=MODEL=AE_2048,ARCH_CONFIG=paper_parse_pbmc runs/scripts/train/st_perturbation.sbatch
#   sbatch --export=MODEL=AE_2048,ARCH_CONFIG=hf_se_parse,DECODER_MODE=frozen runs/scripts/train/st_perturbation.sbatch
#   sbatch --export=MODEL=AE_2048,DECODER_WEIGHT=0.0 runs/scripts/train/st_perturbation.sbatch
#
# Extra-seed run (writes to a parallel root):
#   sbatch --export=MODEL=AE_128,SEED=43,OUT_ROOT=/lustre/.../st_paper_latentsize_extra \
#          runs/scripts/train/st_perturbation.sbatch
#
# Usage (array job):
#   sbatch --array=0-6 runs/scripts/train/st_perturbation.sbatch
#   sbatch --array=0-6 --export=ARCH_CONFIG=paper_parse_pbmc runs/scripts/train/st_perturbation.sbatch
#

#SBATCH --job-name=st_pert
#SBATCH -o logs/slurm/%x_%A_%a_out.txt
#SBATCH -e logs/slurm/%x_%A_%a_err.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=200G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --array=0


# ── resolve MODEL name ──────────────────────────────────────────────
MODELS=(PCA_32)
if [ -z "${MODEL:-}" ]; then
    MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
fi
echo "=== Training ST for model: ${MODEL} ==="

# ── arch config, decoder mode and weight ─────────────────────────────
ARCH_CONFIG=${ARCH_CONFIG:-paper_parse_pbmc}
DECODER_MODE=${DECODER_MODE:-frozen}
DECODER_WEIGHT=${DECODER_WEIGHT:-0.0}
SEED=${SEED:-42}
OUT_ROOT_FLAG=""
if [ -n "${OUT_ROOT:-}" ]; then
    OUT_ROOT_FLAG="--out_root ${OUT_ROOT}"
fi
echo "=== arch_config: ${ARCH_CONFIG}, decoder_mode: ${DECODER_MODE}, decoder_weight: ${DECODER_WEIGHT}, seed: ${SEED}, out_root: ${OUT_ROOT:-default} ==="

# ── environment ─────────────────────────────────────────────────────
EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/.bashrc"
conda deactivate
conda activate cstm_scvi_env
export HYDRA_FULL_ERROR=1

# ── paths ───────────────────────────────────────────────────────────
REPO=${EXPDIR}

mkdir -p "${REPO}/logs/slurm"

# ── scheduler flag (opt-in, default off) ─────────────────────────────
SCHEDULER_FLAG=${SCHEDULER_FLAG:-}  # scheduler off by default; set SCHEDULER_FLAG=--scheduler to enable

# ── train ────────────────────────────────────────────────────────────
python "${REPO}/codes/train_st.py" \
  --model "${MODEL}" \
  --arch_config "${ARCH_CONFIG}" \
  --decoder_mode "${DECODER_MODE}" \
  --max_steps 200000 \
  --batch_size 16 \
  --lr 3e-4 \
  --cell_set_len 512 \
  --decoder_weight "${DECODER_WEIGHT}" \
  --seed "${SEED}" \
  ${OUT_ROOT_FLAG} \
  ${SCHEDULER_FLAG}

echo "=== Done: ${MODEL} ==="
