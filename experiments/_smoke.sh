#!/bin/bash
# Smoke test harness for experiments/ drivers.
#
# Runs each driver with the smallest sensible args and verifies that
# (a) the process exits 0, (b) the expected output file is created.
#
# Drivers are grouped by conda env. Activate the right env first, then run
# this script with the matching --group flag:
#
#   bash _smoke.sh --group cpu              # PCA + STATE-prep on CPU
#   bash _smoke.sh --group e2e_gpu          # AE / scVI / nlscVI on GPU
#   bash _smoke.sh --group fm_se            # SE_emb (needs STATE)
#   bash _smoke.sh --group fm_scgpt         # scGPT_emb
#   bash _smoke.sh --group fm_scconcept     # scConcept_emb
#   bash _smoke.sh --group fm_scimilarity   # scimilarity_emb
#   bash _smoke.sh --group decoder          # decoderonly_hvg + tsfm
#   bash _smoke.sh --group cf               # CellFlow
#   bash _smoke.sh --group st               # STATE
#   bash _smoke.sh --group all              # run every group sequentially
#
# Writes per-driver outputs to ${SMOKE_OUT:-/tmp/reconeval_smoke}/ and
# appends a row to ${SMOKE_OUT}/REPORT.md.

set -uo pipefail

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SMOKE_OUT holds heavy outputs (weights, embeddings). Default /tmp for speed.
SMOKE_OUT="${SMOKE_OUT:-/tmp/reconeval_smoke}"
# SMOKE_DRIVER_LOGS holds per-driver stderr/stdout — defaults to lustre so logs
# survive after the compute node releases /tmp.
SMOKE_DRIVER_LOGS="${SMOKE_DRIVER_LOGS:-$EXPDIR/_smoke_logs/driver_logs}"
# REPORT_DIR keeps the consolidated REPORT.md on lustre as well.
REPORT_DIR="${REPORT_DIR:-$EXPDIR/_smoke_logs}"
mkdir -p "$SMOKE_OUT/logs" "$SMOKE_DRIVER_LOGS" "$REPORT_DIR"
REPORT="$REPORT_DIR/REPORT.md"
export RECONEVAL_OUT="$SMOKE_OUT"
export HYDRA_FULL_ERROR=1

if [[ ! -f "$REPORT" ]]; then
  printf "# Smoke-test report\n\n| Task | Driver | Group | Outcome | Time (s) | Note |\n|---|---|---|---|---|---|\n" > "$REPORT"
fi

GROUP="${1:-}"
if [[ "$GROUP" == "--group" ]]; then GROUP="${2:-all}"; fi
[[ -z "$GROUP" ]] && GROUP="all"

# --- helpers ---------------------------------------------------------------

run_one () {
  # run_one <task> <driver_label> <group> <expected_output_glob> <cmd ...>
  local task="$1" label="$2" group="$3" expect_glob="$4"
  shift 4
  local logfile="$SMOKE_DRIVER_LOGS/${label// /_}.log"
  echo "==[ $task / $label ($group) ]==" | tee -a "$logfile"
  echo "cmd: $*" | tee -a "$logfile"
  local t0 t1 dur outcome note
  t0=$(date +%s)
  if "$@" >>"$logfile" 2>&1; then
    t1=$(date +%s); dur=$((t1-t0))
    if [[ -n "$expect_glob" ]] && ! compgen -G "$expect_glob" >/dev/null; then
      outcome="FAIL"; note="exit 0 but no output matching $expect_glob"
    else
      outcome="PASS"; note="$( [[ -n "$expect_glob" ]] && echo "$expect_glob" || echo "exit 0")"
    fi
  else
    t1=$(date +%s); dur=$((t1-t0))
    outcome="FAIL"; note="exit nonzero; see $logfile"
  fi
  echo "→ $outcome  ($dur s)  $note" | tee -a "$logfile"
  printf "| %s | %s | %s | %s | %d | %s |\n" "$task" "$label" "$group" "$outcome" "$dur" "$note" >> "$REPORT"
}

skip_one () {
  # skip_one <task> <driver_label> <group> <reason>
  printf "| %s | %s | %s | SKIP | 0 | %s |\n" "$1" "$2" "$3" "$4" >> "$REPORT"
  echo "==[ $1 / $2 ($3) ]==  SKIP: $4"
}

want () { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }

# --- 01_end_to_end ---------------------------------------------------------

if want cpu; then
  # 01_end_to_end has no CPU-only driver: PCA uses rapids_singlecell (cuPCA),
  # and the neural models require CUDA. The only CPU-side driver is the data
  # prep utility — covered in the `extras` group at the bottom.
  echo "(no CPU smoke targets in 01_end_to_end — PCA and neural models all need GPU)"
fi

if want e2e_gpu; then
  # PCA writes a directory (mean + pc files), not a single .pt — match the dir.
  run_one "01_e2e" "PCA d=10 pbmc" "e2e_gpu" \
    "$SMOKE_OUT/weights/pbmc/split03/PCA/Default" \
    python "$EXPDIR/01_end_to_end/codes/train.py" \
      model=train/PCA data=pbmc trainer=PCA \
      model.model_args.n_components=10 split=split03 \
      data.model_specific.PCA.chunk_size=50000
  # Neural — limit_*_batches keeps the epoch tiny but still triggers
  # validation so val/loss_epoch is emitted (EarlyStopping needs it).
  # num_workers is not in the data config — use Hydra '+' to add it.
  for m in AE scVI nlscVI mlscVI; do
    if [[ "$m" == "AE" ]]; then trainer_cfg=AE; else trainer_cfg=scVI; fi
    run_one "01_e2e" "$m d=10 pbmc" "e2e_gpu" \
      "$SMOKE_OUT/weights/pbmc/split03/$m/*/*" \
      python "$EXPDIR/01_end_to_end/codes/train.py" \
        "model=train/$m" data=pbmc "trainer=$trainer_cfg" \
        model.model_args.n_latent=10 split=split03 \
        trainer.max_epochs=1 trainer.min_epochs=1 \
        +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 \
        "+data.model_specific.$m.num_workers=1" \
        "data.model_specific.$m.minibatch_size=128"
  done
fi

# --- 02_foundation_model — embed side -------------------------------------

# Each FM embed needs its own env; the user activates it before launching.
fm_smoke_run () {
  # fm_smoke_run <FM> <group_tag>
  # total_parts=17000 parts=1 → only 1 (out of ~17030) combination is processed:
  # a tiny slice that still exercises model load + forward + zarr write.
  local fm="$1" group="$2"
  local out="$SMOKE_OUT/${fm}_emb.zarr"
  rm -rf "$out"
  run_one "02_fm" "${fm}_emb" "$group" \
    "$out" \
    python "$EXPDIR/02_foundation_model/codes/${fm}_emb.py" \
      total_parts=17000 parts=1 \
      "data_args.output_dir=$out"
}

want fm_se          && fm_smoke_run SE          fm_se
want fm_scgpt       && fm_smoke_run scGPT       fm_scgpt
want fm_scconcept   && fm_smoke_run scConcept   fm_scconcept
want fm_scimilarity && fm_smoke_run scimilarity fm_scimilarity

# --- 02_foundation_model — decoder side -----------------------------------

if want decoder; then
  # weights.dir = ${RECONEVAL_OUT}/decoder/${data}/${gene_space}/${latent_key}/${split}/${decoder_name}/
  # MLP smoke runs in the SE env (cstm_scvi_env).
  run_one "02_fm" "decoderonly_hvg" "decoder" \
    "$SMOKE_OUT/decoder/pbmc/hvg/SE/split02/MLP/*/*" \
    python "$EXPDIR/02_foundation_model/codes/decoderonly_hvg.py" \
      pretrained=SE decoder=MLP data=pbmc \
      decoder.max_epochs=1 +decoder.max_steps=2
fi

# Transformer decoder needs `concept` — run it under scconcept_env.
# decoder/Transformer.yaml defaults to 2-GPU DDP; force single-GPU for smoke.
if want decoder_tsfm; then
  run_one "02_fm" "decoderonly_hvg_tsfm" "decoder_tsfm" \
    "$SMOKE_OUT/decoder/pbmc/hvg/SE/split02/Transformer/*/*" \
    python "$EXPDIR/02_foundation_model/codes/decoderonly_hvg_tsfm.py" \
      pretrained=SE decoder=Transformer data=pbmc \
      decoder.max_epochs=1 +decoder.max_steps=2 \
      decoder.ddp.devices=1 decoder.ddp.strategy=auto
fi

# --- 03_latent_shift -------------------------------------------------------

if want cf; then
  # cellflow checkpoints pickle the whole model (~hundreds of MB). Write to
  # the lustre-backed log dir so we don't fill node-local /tmp.
  cf_out="$REPORT_DIR/cf_smoke_${SLURM_JOB_ID:-local}"
  rm -rf "$cf_out"
  run_one "03_ls" "train_cf PCA_128" "cf" \
    "$cf_out/PCA_128*" \
    python "$EXPDIR/03_latent_shift/codes/train_cf.py" \
      --model PCA_128 --num_iters 100 --batch_size 64 --valid_freq 100 \
      --out_dir "$cf_out"
fi

if want st; then
  run_one "03_ls" "train_st PCA_128" "st" \
    "$SMOKE_OUT/st_smoke/*" \
    python "$EXPDIR/03_latent_shift/codes/train_st.py" \
      --model PCA_128 --max_steps 100 --batch_size 4 --no_wandb \
      --out_root "$SMOKE_OUT/st_smoke"
fi

# --- always-skipped (heavy data dependencies) -----------------------------

if [[ "$GROUP" == "all" || "$GROUP" == "extras" ]]; then
  skip_one "01_e2e" "extract_e2e_embeddings.py" "extras" "no CLI; needs trained checkpoint + paper zarr"
  skip_one "03_ls"  "prepare_pbmc_st.py"        "extras" "argparse but needs paper PBMC zarr; covered by train_cf/train_st"
fi

echo
echo "=== Report so far ==="
cat "$REPORT"
