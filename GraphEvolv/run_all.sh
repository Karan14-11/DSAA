#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Master script to run DyCond on all 5 datasets × 3 loss types
#
# Usage:
#   bash run_all.sh              # Run everything
#   bash run_all.sh --small      # Run only small datasets first
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

EPOCHS=1000
SAVE_EVERY=250
BATCH_SIZE=4
LR=1e-4

# Datasets ordered by size (smallest first)
SMALL_DATASETS="cit_hepph"
LARGE_DATASETS="euroroad cit_hepph pp_pathways collegemsg bitcoin_alpha"
LOSS_TYPES="deg_slope"

if [ "$1" == "--small" ]; then
    DATASETS="$SMALL_DATASETS"
    echo "=== Running SMALL datasets only ==="
else
    DATASETS="$LARGE_DATASETS"
    echo "=== Running ALL datasets ==="
fi


echo ""
echo "Datasets: $DATASETS"
echo "Loss types: $LOSS_TYPES"
echo "Epochs: $EPOCHS"
echo ""

# ── Step 1: Preprocess ─────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "STEP 1: Preprocessing all datasets"
echo "═══════════════════════════════════════════════════"


# ── Step 2: Train ──────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "STEP 2: Training (${EPOCHS} epochs each)"
echo "═══════════════════════════════════════════════════"

for DATASET in $DATASETS; do
    for LOSS in $LOSS_TYPES; do
        echo ""
        echo ">>> Training: $DATASET | $LOSS"
        echo "    Log: logs/${DATASET}_${LOSS}.log"

        mkdir -p logs

        python3 train_multi_dataset.py \
            --dataset "$DATASET" \
            --dycond_loss_type "$LOSS" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --save_every "$SAVE_EVERY" \
            2>&1 | tee "logs/${DATASET}_${LOSS}.log"
    done
done

# ── Step 3: Inference ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "STEP 3: Generating graph timelines"
echo "═══════════════════════════════════════════════════"

for DATASET in $DATASETS; do
    for LOSS in $LOSS_TYPES; do
        echo ""
        echo ">>> Inference: $DATASET | $LOSS"

        python3 infer_multi_dataset.py \
            --dataset "$DATASET" \
            --dycond_loss_type "$LOSS" \
            --checkpoints 250 500 750 1000 best \
            --steps 50 \
            --candidates 20000
    done
done

# ── Step 4: Evaluate ───────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "STEP 4: Evaluating generated graphs"
echo "═══════════════════════════════════════════════════"

python3 evaluate_all.py --epochs "250,500,750,1000,best" 

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ ALL DONE!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Results saved in: results/<dataset>/<loss_type>/"
echo "Logs saved in: logs/"
echo "Checkpoints in: checkpoints/<dataset>_<loss_type>/"
