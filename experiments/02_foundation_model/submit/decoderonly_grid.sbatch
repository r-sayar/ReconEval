#!/bin/bash

#SBATCH -J de_mlp_grid
#SBATCH -o logs/slurm/%x_out.out
#SBATCH -e logs/slurm/%x_err.err

#SBATCH -p cpu_p
#SBATCH --qos=cpu_priority
#SBATCH --mem=20G

#SBATCH -t 0-01:00:00





EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source $HOME/.bashrc
conda deactivate
conda activate scconcept_env
export HYDRA_FULL_ERROR=1

python ${EXPDIR}/codes/decoderonly_hvg_tsfm.py \
    --multirun \
    model=eval/AE \
    decoder=MLP \
    data=pbmc \
    decoder.model_args.n_hidden=4096 \
    decoder.model_args.n_layers=2 \
    split=split03\
    pretrained=scConcept \
    gene_space=hvg \
    gene_dim=5412 \
    data_args.minibatch_size=256 \
    hvg.all_genes_zarr='/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc_w_ag/comb_w_obs.zarr' \
    hvg.target_genes_zarr='/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/pbmc/comb_w_obs.zarr' \
    +launcher=gpu_normal_constraint_reduced