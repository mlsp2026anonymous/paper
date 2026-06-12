#!/bin/bash
# Sweep w_0 from 0.00 to 1.00 for a fixed SF-UniDA setup.

set -euo pipefail

trap 'echo ""; echo "Interrupted — exiting."; exit 130' INT TERM

DATA_DIR="data"
SAVE_DIR="logs_threshold_sensitivity"
BASE_HPARAMS_JSON="configs/best_hparams/sfunida/best_hparams_CNN.json"
TMPDIR_HPARAMS="/tmp/thr_sensitivity_hparams"

METHOD="GLC"
DATASET="HAR"
BACKBONES=(CNN FNO S3Layer Mantis Chronos)

mkdir -p "$TMPDIR_HPARAMS"

total=$(( ${#BACKBONES[@]} * 21 ))
count=0

for BACKBONE in "${BACKBONES[@]}"; do
    for THR in $(LC_NUMERIC=C seq -f "%.2f" 0 0.05 1.00); do
        count=$(( count + 1 ))
        EXP_NAME="THR_${THR}"
        PATCHED_JSON="$TMPDIR_HPARAMS/hparams_${BACKBONE}_${THR}.json"

        echo ""
        echo "[$count/$total] w_0=$THR | $DATASET | $BACKBONE | $METHOD"

        python scripts/conda/patch_hparams.py \
            --input   "$BASE_HPARAMS_JSON" \
            --output  "$PATCHED_JSON"      \
            --dataset "$DATASET"           \
            --method  "$METHOD"            \
            --key     w_0                  \
            --value   "$THR"

        python main.py \
            --dataset      "$DATASET"      \
            --backbone     "$BACKBONE"     \
            --da_method    "$METHOD"       \
            --data_path    "$DATA_DIR"     \
            --save_dir     "$SAVE_DIR"     \
            --exp_name     "$EXP_NAME"     \
            --num_runs     10              \
            --hparams_json "$PATCHED_JSON"
    done
done

rm -rf "$TMPDIR_HPARAMS"

echo ""
echo "Threshold sensitivity done."
