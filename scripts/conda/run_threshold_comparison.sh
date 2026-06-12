#!/bin/bash
# Compare auto-thresholding methods for SF-UniDA with the current Python environment.

set -uo pipefail

trap 'echo ""; echo "Interrupted — exiting."; exit 130' INT TERM

DATA_DIR="data"
HPARAMS_DIR="configs/best_hparams/sfunida"
SAVE_DIR="logs_threshold_comparison"
TABLES_DIR="tables/threshold_comparison"

METHODS=("GLC")
DATASETS=("HAR")
BACKBONES=(
    CNN
    FNO
    TSLANet
    S3Layer
    Mantis
    Chronos
    Moment
)
THR_METHODS=("yen" "otsu" "li")

total=$(( ${#DATASETS[@]} * ${#BACKBONES[@]} * ${#METHODS[@]} * ${#THR_METHODS[@]} ))
count=0

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for THR in "${THR_METHODS[@]}"; do
                count=$(( count + 1 ))
                echo ""
                echo "[$count/$total] $DATASET | $BACKBONE | $METHOD | threshold=$THR"

                HPARAMS_FILE="${HPARAMS_DIR}/best_hparams_${BACKBONE}.json"
                THR_SAVE_DIR="${SAVE_DIR}/${THR}"

                python main.py \
                    --dataset          "$DATASET"       \
                    --backbone         "$BACKBONE"      \
                    --da_method        "$METHOD"        \
                    --data_path        "$DATA_DIR"      \
                    --save_dir         "$THR_SAVE_DIR"  \
                    --exp_name         THR_EXP1         \
                    --num_runs         3                \
                    --auto_threshold                    \
                    --threshold_method "$THR"           \
                    --hparams_json     "$HPARAMS_FILE"
            done
        done
    done
done

mkdir -p "$TABLES_DIR"

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        python scripts/tables/generate_threshold_table.py \
            --root        "$SAVE_DIR"                         \
            --method      "$METHOD"                           \
            --dataset     "$DATASET"                          \
            --backbones   "${BACKBONES[@]}"                   \
            --thr-methods "${THR_METHODS[@]}"                 \
            --exp-name    THR_EXP1                            \
            --metric      "H-score"                           \
            --output      "$TABLES_DIR/thr_${METHOD}_${DATASET}.tex"
    done
done

python scripts/tables/compile_tables.py \
    --tables-dir "$TABLES_DIR" \
    --force

echo ""
echo "Threshold comparison done."
