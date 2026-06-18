#!/bin/bash
#SBATCH --job-name=split_comb
#SBATCH --output=logs/slurm/split_comb_%j.out
#SBATCH --error=logs/slurm/split_comb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=300G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

source $HOME/.bashrc
conda deactivate
conda activate scvi_env

# Define input and output paths
SOURCE_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/comb_w_obs"
OUTPUT_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/final/tahoe/split03"

# Set parameters
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15
SEED=42

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the combination split
echo "Starting combination split..."
python -m /lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/runs/handler/data/_process_tahoe/split_comb.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SEED"

echo "Combination split complete" 