#!/bin/bash

set -uo pipefail

ENV_NAME="${1:-sf-unidabench}"

conda create -n "$ENV_NAME" python=3.11 -y
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install --no-deps -r requirements_no_deps.txt

echo ""
echo "Environment ready. Activate it with:"
echo "  conda activate $ENV_NAME"
