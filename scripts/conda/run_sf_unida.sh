#!/bin/bash
# Run SF-UniDA method × backbone × dataset combinations with the current Python environment.

set -uo pipefail

trap 'echo ""; echo "Interrupted — exiting."; exit 130' INT TERM


DATA_DIR="data"
HPARAMS_DIR="configs/best_hparams/sfunida"
SF_UNIDA_AUTO_SAVE_DIR="logs_test_mlsp26/run_sf_unida_auto"
SF_UNIDA_SAVE_DIR="logs_test_mlsp26/run_sf_unida"
THRESHOLD_METHOD="yen"

METHODS=("GLC" "GLCpp" "LEAD" "UMAD" "UniJDOT")
DATASETS=("HAR" "HHAR" "EEG")
BACKBONES=(
    CNN
    FNO
    TSLANet
    S3Layer
    Mantis
    Chronos
    Moment
)

total=$(( ${#DATASETS[@]} * ${#BACKBONES[@]} * ${#METHODS[@]} ))
count=0

#mkdir -p "$SF_UNIDA_AUTO_SAVE_DIR"

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            count=$(( count + 1 ))
            echo ""
            echo "[$count/$total] auto-threshold | $DATASET | $BACKBONE | $METHOD"

            HPARAMS_ARGS=()
            HPARAMS_FILE="${HPARAMS_DIR}/best_hparams_${BACKBONE}.json"

            python main.py \
                --dataset          "$DATASET"                 \
                --backbone         "$BACKBONE"                \
                --da_method        "$METHOD"                  \
                --data_path        "$DATA_DIR"                \
                --save_dir         "$SF_UNIDA_AUTO_SAVE_DIR"  \
                --exp_name         EXP1                       \
                --num_runs         10                         \
                --auto_threshold                              \
                --threshold_method "$THRESHOLD_METHOD"        \
                --hparams_json "$HPARAMS_FILE"
        done
    done
done

count=0
#mkdir -p "$SF_UNIDA_SAVE_DIR"

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            count=$(( count + 1 ))
            echo ""
            echo "[$count/$total] fixed-threshold | $DATASET | $BACKBONE | $METHOD"

            HPARAMS_FILE="${HPARAMS_DIR}/best_hparams_${BACKBONE}.json"

            python main.py \
                --dataset   "$DATASET"        \
                --backbone  "$BACKBONE"       \
                --da_method "$METHOD"         \
                --data_path "$DATA_DIR"       \
                --save_dir  "$SF_UNIDA_SAVE_DIR" \
                --exp_name  EXP1              \
                --num_runs  10                \
                --hparams_json "$HPARAMS_FILE"
        done
    done
done

echo ""
echo "SF-UniDA runs done."
