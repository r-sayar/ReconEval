#!/bin/bash
# Latent-size sweep for end-to-end reconstruction on PBMC.
#
# Submits one job per (model, latent) cell of the grid by re-invoking the
# single-model sbatch wrappers. Override via env vars:
#
#   MODELS    space-separated model names from {PCA, AE, scVI, nlscVI, mlscVI}
#             default: "PCA AE scVI nlscVI mlscVI"
#   LATENTS   space-separated latent dims
#             default: "10 32 128 512 2048"
#   DATA      hydra data config name (data/<DATA>.yaml)
#             default: pbmc
#   SPLIT     split tag passed through to train.py via Hydra
#             default: split03
#   MAX_EPOCHS / MIN_EPOCHS  trainer caps
#             defaults: 400 / 10
#
# Example:
#   MODELS="AE scVI" LATENTS="10 128" bash pbmc_search.sh

set -euo pipefail

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-PCA AE scVI nlscVI mlscVI}"
LATENTS="${LATENTS:-10 32 128 512 2048}"
DATA="${DATA:-pbmc}"
SPLIT="${SPLIT:-split03}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"

for model in $MODELS; do
  case "$model" in
    PCA)    sbatch_file="${EXPDIR}/submit/train_pca.sbatch";  latent_key=n_components ;;
    AE|scVI|nlscVI|mlscVI)
            sbatch_file="${EXPDIR}/submit/train_${model,,}.sbatch"
            # scVI sbatch is reused for nlscVI/mlscVI (same trainer, MODEL override)
            case "$model" in
              nlscVI|mlscVI) sbatch_file="${EXPDIR}/submit/train_scvi.sbatch" ;;
            esac
            latent_key=n_latent ;;
    *) echo "unknown model $model"; exit 1 ;;
  esac
  for latent in $LATENTS; do
    jobname="${model,,}_${DATA}_${SPLIT}_d${latent}"
    echo "submitting $jobname"
    sbatch --job-name="$jobname" \
           --export=ALL,MODEL="$model",LATENT="$latent",DATA="$DATA",SPLIT="$SPLIT",MAX_EPOCHS="$MAX_EPOCHS",MIN_EPOCHS="$MIN_EPOCHS",LATENT_KEY="$latent_key" \
           "$sbatch_file"
  done
done
