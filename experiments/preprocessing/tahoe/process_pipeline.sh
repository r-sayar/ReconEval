#!/bin/bash
# Controller for reconstruction data handling pipeline
# Steps:
# 1. Save the source dataset to a h5ad file - normalizing, log1p, hvg selection.
# 2. Read h5ad file, save the atom unit as zarr file for evalutation - split key: cell line X drug X concentration, with attributes of combination key and gene names
# 3. Split the combined zarr into sub zarrs based on different granularity - split01,02,03 

set -euo pipefail


DATASET_NAME="${DATASET_NAME:-tahoe}"
STEPS="${STEPS:-split}"      
SPLITS="${SPLITS:-comb}"     
NUM_SPLITS=15

LOG_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/logs/${DATASET_NAME}"
mkdir -p "${LOG_DIR}"

SAVE_SOURCE_FILE="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/raw_counts/tahoe_counts_3000wCCM.h5ad"
COMB_SOURCE_FILE="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe_counts/comb_w_obs_3000wCCM"
SPLIT_OUTPUT_BASE="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/tahoe_counts"
SPLIT_RATIOS="${SPLIT_RATIOS:-0.7 0.15 0.15}"
SEED="${SEED:-42}"

SAVE_SCRIPT="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/runs/handler/data/_process_tahoe/tahoe_saveonly.py"
COMB_SCRIPT="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/runs/handler/data/_process_tahoe/save_comb.py"
MERGE_SCRIPT="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/runs/handler/data/_process_tahoe/merge_mcomb.py"
SPLIT_SCRIPTS_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/runs/handler/data/_process_tahoe/_split_NB"

# Conda envs
SAVE_ENV="${SAVE_ENV:-cstm_scvi_env}"
COMB_ENV="${COMB_ENV:-cstm_scvi_env}"
MERGE_ENV="${MERGE_ENV:-cstm_scvi_env}"
SPLIT_ENV="${SPLIT_ENV:-cstm_scvi_env}"


have_step() {
  [[ ",${STEPS}," == *",$1,"* ]]
}

IFS=',' read -r -a SPLIT_TYPES <<< "${SPLITS}"

train_ratio=$(echo ${SPLIT_RATIOS} | cut -d' ' -f1)
val_ratio=$(echo   ${SPLIT_RATIOS} | cut -d' ' -f2)
test_ratio=$(echo  ${SPLIT_RATIOS} | cut -d' ' -f3)

echo "Dataset     : ${DATASET_NAME}"
echo "Steps       : ${STEPS}"
echo "Split types : ${SPLITS}"
echo "Num splits  : ${NUM_SPLITS}"
echo "Ratios      : train=${train_ratio} val=${val_ratio} test=${test_ratio}"
echo




SAVE_JOB=""
COMB_JOBS=()  
MERGE_JOB=""
SPLIT_JOBS=()

# --- SAVE ---
if have_step save; then
  echo "Submitting SAVE job..."
  SAVE_JOB=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=save_${DATASET_NAME}
#SBATCH -o ${LOG_DIR}/save_%j.out
#SBATCH -e ${LOG_DIR}/save_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

source \$HOME/.bashrc
conda activate ${SAVE_ENV}
python ${SAVE_SCRIPT} \
  --outputpath "${SAVE_SOURCE_FILE}" 
EOF
)
  echo "  SAVE_JOB=${SAVE_JOB}"
else
  # If skipping save, ensure input exists
  if [[ ! -f "${SAVE_SOURCE_FILE}" ]]; then
    echo "ERROR: SAVE step skipped but h5ad not found at ${SAVE_SOURCE_FILE}"
    exit 1
  fi
fi



# --- COMB (Parallel) ---
if have_step comb; then
  echo "Submitting ${NUM_SPLITS} parallel COMB jobs..."
  dep_opt=""
  if [[ -n "${SAVE_JOB}" ]]; then
    dep_opt="--dependency=afterok:${SAVE_JOB}"
  fi
  
  for ((i=0; i<NUM_SPLITS; i++)); do
    JOB_ID=$(sbatch --parsable ${dep_opt} <<EOF
#!/bin/bash
#SBATCH --job-name=save_comb_${i}
#SBATCH -o ${LOG_DIR}/save_comb_${i}_%j.out
#SBATCH -e ${LOG_DIR}/save_comb_${i}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

source \$HOME/.bashrc
conda activate ${COMB_ENV}
python ${COMB_SCRIPT} \
  --h5-path "${SAVE_SOURCE_FILE}" \
  --output-dir "${COMB_SOURCE_FILE}" \
  --part ${i} \
  --total-parts ${NUM_SPLITS}
EOF
)
    COMB_JOBS+=("${JOB_ID}")
    echo "  COMB_${i} -> ${JOB_ID}"
  done
fi

# --- MERGE ---
if have_step merge; then
  echo "Submitting MERGE job..."
  dep_opt=""
  if [[ ${#COMB_JOBS[@]} -gt 0 ]]; then
    dep_opt="--dependency=afterok:$(IFS=:; echo "${COMB_JOBS[*]}")"
  fi
  
  MERGE_JOB=$(sbatch --parsable ${dep_opt} <<EOF
#!/bin/bash
#SBATCH --job-name=merge_comb
#SBATCH -o ${LOG_DIR}/merge_comb_%j.out
#SBATCH -e ${LOG_DIR}/merge_comb_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority

source \$HOME/.bashrc
conda activate ${MERGE_ENV}
python ${MERGE_SCRIPT} \
  --output-dir "${COMB_SOURCE_FILE}" \
  --total-parts ${NUM_SPLITS} \
  --workers 28 
EOF
)
  echo "  MERGE_JOB=${MERGE_JOB}"
else
  # If skipping merge, ensure output exists
  if [[ ! -d "${COMB_SOURCE_FILE}" ]]; then
    echo "ERROR: MERGE step skipped but zarr source dir not found at ${COMB_SOURCE_FILE}"
  fi
fi


# --- SPLIT ---
if have_step split; then
  echo "Submitting SPLIT jobs..."
  for SPLIT_TYPE in "${SPLIT_TYPES[@]}"; do
    case $SPLIT_TYPE in
      cell_line) SPLIT_INDEX=1 ;;
      drug)      SPLIT_INDEX=2 ;;
      comb)      SPLIT_INDEX=3 ;;
      *) echo "ERROR: unknown split type '${SPLIT_TYPE}'"; exit 1 ;;
    esac

    dep_opt=""
    if [[ -n "${MERGE_JOB}" ]]; then
      dep_opt="--dependency=afterok:${MERGE_JOB}"
    fi

    JOB_ID=$(sbatch --parsable ${dep_opt} <<EOF
#!/bin/bash
#SBATCH --job-name=split_${SPLIT_TYPE}
#SBATCH --output=${LOG_DIR}/split_${SPLIT_TYPE}_%j.out
#SBATCH --error=${LOG_DIR}/split_${SPLIT_TYPE}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

source \$HOME/.bashrc
conda activate ${SPLIT_ENV}

python ${SPLIT_SCRIPTS_DIR}/split_${SPLIT_TYPE}.py \
  --source-dir "${COMB_SOURCE_FILE}" \
  --output-dir "${SPLIT_OUTPUT_BASE}/split0${SPLIT_INDEX}" \
  --train-ratio ${train_ratio} \
  --val-ratio ${val_ratio} \
  --test-ratio ${test_ratio} \
  --seed ${SEED}
EOF
)
    SPLIT_JOBS+=("${JOB_ID}")
    echo "  split_${SPLIT_TYPE} -> ${JOB_ID}"
  done
fi

echo
echo "Pipeline submitted:"
[[ -n "${SAVE_JOB}"  ]] && echo "  SAVE : ${SAVE_JOB}"
((${#COMB_JOBS[@]})) && echo "  COMB: ${COMB_JOBS[*]}"
[[ -n "${MERGE_JOB}" ]] && echo "  MERGE: ${MERGE_JOB}"
((${#SPLIT_JOBS[@]})) && echo "  SPLIT: ${SPLIT_JOBS[*]}"