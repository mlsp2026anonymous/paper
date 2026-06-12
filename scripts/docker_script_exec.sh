#!/bin/bash

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 path/to/script.sh [script args...]"
    exit 2
fi

SCRIPT_PATH="$1"
shift

CONTAINER_NAME="sf_unidabench_exec_${USER:-user}_$$"

cleanup_container() {
    if [ -n "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")" ]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}

on_exit() {
    exit_code=$?
    trap - EXIT INT TERM HUP
    cleanup_container
    exit "$exit_code"
}

trap on_exit EXIT INT TERM HUP

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --shm-size=200g \
    -v "$PWD:/workspace" \
    -w /workspace \
    -e HF_HOME=/workspace/.hf_cache \
    sf_unidabench:latest \
    bash "$SCRIPT_PATH" "$@" &

DOCKER_RUN_PID=$!
wait "$DOCKER_RUN_PID"
