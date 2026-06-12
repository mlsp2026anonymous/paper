#!/bin/bash
# pretrain_all_source_models.sh — Pretrain source models with the current Python environment.
#
# Usage:
#   ./scripts/conda/pretrain_all_source_models.sh
#   DATA_DIR=/custom/path ./scripts/conda/pretrain_all_source_models.sh
#
# Run in tmux to survive SSH disconnects:
#   tmux new-session -s pretrain './scripts/conda/pretrain_all_source_models.sh'

set -uo pipefail

trap 'echo ""; log "Interrupted — exiting."; exit 130' INT TERM


DATA_DIR="data"
DEVICE="cuda"
DATASETS=(HAR HHAR EEG)


BACKBONES=(
    CNN
    FNO
    TSLANet
    S3Layer
    Mantis
    Chronos
    Moment
)

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do

        python pretrain_source_models.py \
            --backbone  "$BACKBONE" \
            --dataset   "$DATASET"  \
            --data_path "$DATA_DIR" \
            --device    "$DEVICE"   \
            --force
    done
done

log "All source models pretrained."
