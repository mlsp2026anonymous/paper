#!/bin/bash
# Generate tables using results2.csv (mean ± std across seeds in the mean row).
# Run recompute_results.py first to produce results2.csv from results_temp.csv.

set -uo pipefail

PYTHON_BIN="python"
EXP_NAME="EXP1"

ROOT="logs_test_mlsp26/run_sf_unida_auto"
OUT="tables/"
SCRIPT_DIR="scripts/tables"
mkdir -p "$OUT"

# ── Step 1: recompute results2.csv from results_temp.csv ─────────────────────
python "$SCRIPT_DIR/recompute_results.py" --root "$ROOT"

# ── Step 2: summary table (all datasets × all backbones) with Auto-Threshold ─────────────────────
python "$SCRIPT_DIR/generate_summary_table.py" \
    --root           "$ROOT"                              \
    --metric         H-score                              \
    --datasets       HAR HHAR EEG                         \
    --methods        UniJDOT UMAD GLC GLCpp LEAD                  \
    --method-names   UniJDOT UMAD GLC "GLC++" LEAD                \
    --backbones      CNN FNO S3Layer TSLANet Mantis Moment Chronos \
    --backbone-names CNN FNO S3 TSLANet Mantis Moment Chronos \
    --ext            "EXP1"                          \
    --output         "$OUT/summary_sf_unida_hscore_auto.tex"

# ── Step 3: summary table (all datasets × all backbones) without Auto-Threshold ─────────────────────

ROOT="logs_test_mlsp26/run_sf_unida"
python "$SCRIPT_DIR/recompute_results.py" --root "$ROOT"
python "$SCRIPT_DIR/generate_summary_table.py" \
    --root           "$ROOT"                              \
    --metric         H-score                              \
    --datasets       HAR HHAR EEG                         \
    --methods        UniJDOT UMAD GLC GLCpp LEAD                  \
    --method-names   UniJDOT UMAD GLC "GLC++" LEAD                \
    --backbones      CNN FNO S3Layer TSLANet Mantis Moment Chronos \
    --backbone-names CNN FNO S3 TSLANet Mantis Moment Chronos \
    --ext            "EXP1"                          \
    --output         "$OUT/summary_sf_unida_hscore.tex"


# ── Step 4: comparison table with/without Auto-Threshold ─────────────────────

python "$SCRIPT_DIR/compare_latex_tables.py" "$OUT/summary_sf_unida_hscore_auto.tex" "$OUT/summary_sf_unida_hscore.tex" --color --star-unijdot


# ── Step 5: compile all .tex to PNG ──────────────────────────────────────────
python "$SCRIPT_DIR/compile_tables.py" --tables-dir "$OUT" --force
