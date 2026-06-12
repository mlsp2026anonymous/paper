#!/bin/bash
# pretrain_all_source_models_umad.sh — Pretrain UMAD two-head source models for all backbone × dataset combos.
#
# Runs pretrain_source_models_umad.py for every combination with the current
# Python environment.
# Existing checkpoints are skipped. HP is loaded from cache or grid-searched if
# not yet cached.
#
# Usage:
#   ./scripts/conda/pretrain_all_source_models_umad.sh
#   DATA_DIR=/custom/path/data ./scripts/conda/pretrain_all_source_models_umad.sh
#
# Run in tmux to survive SSH disconnects:
#   tmux new-session -s pretrain_umad './scripts/conda/pretrain_all_source_models_umad.sh'

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

        python pretrain_source_models_umad.py \
            --backbone  "$BACKBONE" \
            --dataset   "$DATASET"  \
            --data_path "$DATA_DIR" \
            --device    "$DEVICE"
    done
done

log "All UMAD source models pretrained."
