#!/bin/bash

#SBATCH --job-name=tr_pca
#SBATCH -o logs/slurm/%x_%j_out.txt
#SBATCH -e logs/slurm/%x_%j_err.txt

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=2:00:00
#SBATCH --nice=0

# Env-var overrides honoured by pbmc_search.sh / direct sbatch --export:
#   DATA   (default: tahoe)
#   SPLIT  (default: split03)
#   LATENT (default: 128)  → PCA n_components

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source $HOME/.bashrc
conda deactivate
conda activate cstm_scvi_env
export HYDRA_FULL_ERROR=1

python ${EXPDIR}/codes/train.py \
  model=train/PCA \
  data=${DATA:-tahoe} \
  trainer=PCA \
  model.model_args.n_components=${LATENT:-128} \
  split=${SPLIT:-split03}
