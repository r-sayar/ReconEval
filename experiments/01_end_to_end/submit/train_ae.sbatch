#!/bin/bash

#SBATCH --job-name=tr_ae
#SBATCH -o logs/slurm/%x_%j_out.txt  
#SBATCH -e logs/slurm/%x_%j_err.txt         

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --partition=gpu_p    
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb"
#SBATCH --time=24:00:00
#SBATCH --nice=0


EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source $HOME/.bashrc
conda deactivate
conda activate cstm_scvi_env
export HYDRA_FULL_ERROR=1

python ${EXPDIR}/codes/train.py \
    model=train/AE \
    data=${DATA:-tahoe} \
    trainer=AE \
    model.model_args.n_latent=${LATENT:-128} \
    trainer.max_epochs=${MAX_EPOCHS:-400} \
    trainer.min_epochs=${MIN_EPOCHS:-10}

