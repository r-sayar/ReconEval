#!/bin/bash

#SBATCH --job-name=tr_scvi
#SBATCH -o logs/slurm/%x_%j_out.txt
#SBATCH -e logs/slurm/%x_%j_err.txt

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40gb"
#SBATCH --time=24:00:00
#SBATCH --nice=0

# Env-var overrides honoured by pbmc_search.sh / direct sbatch --export:
#   MODEL  (default: scVI; also accepts nlscVI to reuse this script)
#   DATA   (default: tahoe)
#   SPLIT  (default: split03)
#   LATENT (default: 128)  → n_latent
#   N_HIDDEN (default: 1024)
#   MAX_EPOCHS (default: 200)
#   MIN_EPOCHS (default: 1)

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source $HOME/.bashrc
conda deactivate
conda activate cstm_scvi_env
export HYDRA_FULL_ERROR=1

python ${EXPDIR}/codes/train.py \
    model=train/${MODEL:-scVI} \
    data=${DATA:-tahoe} \
    trainer=scVI \
    model.model_args.n_hidden=${N_HIDDEN:-1024} \
    model.model_args.n_latent=${LATENT:-128} \
    split=${SPLIT:-split03} \
    trainer.max_epochs=${MAX_EPOCHS:-200} \
    trainer.min_epochs=${MIN_EPOCHS:-1}
