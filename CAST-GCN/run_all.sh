#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Master script: Link Prediction with Incremental GCN
#
# Runs the full experiment pipeline for all datasets:
#   1. Preprocess raw temporal edges → compact .pt files
#   2. Train & evaluate all 6 strategies on each dataset
#
# Usage:
#   bash run_all.sh                  # Run all datasets
#   bash run_all.sh mathoverflow     # Run single dataset
#   bash run_all.sh --small          # Run small datasets only
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

# ── Dataset lists ─────────────────────────────────────────────
# Ordered by size (smallest first)
SMALL_DATASETS="mathoverflow askubuntu"
LARGE_DATASETS="superuser wikitalk stackoverflow"
ALL_DATASETS="$SMALL_DATASETS $LARGE_DATASETS"

# ── Parse arguments ───────────────────────────────────────────
if [ "$1" == "--small" ]; then
    DATASETS="$SMALL_DATASETS"
    echo "=== Running SMALL datasets only ==="
elif [ -n "$1" ]; then
    DATASETS="$1"
    echo "=== Running single dataset: $1 ==="
else
    DATASETS="$ALL_DATASETS"
    echo "=== Running ALL datasets ==="
fi

echo ""
echo "Datasets: $DATASETS"
echo ""

# ── Step 1: Preprocess ────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "STEP 1: Preprocessing raw datasets"
echo "═══════════════════════════════════════════════════"

# python3 preprocess_dataset.py --all

# ── Step 2: Train & Evaluate ──────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "STEP 2: Running experiments"
echo "═══════════════════════════════════════════════════"

for DATASET in $DATASETS; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ">>> Dataset: $DATASET"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python3 src/train_snapshot.py --dataset "$DATASET" --reverse
done

# ── Step 3: Summary ───────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ ALL DONE!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Results saved in:"
for DATASET in $DATASETS; do
    echo "  results/$DATASET/"
done
