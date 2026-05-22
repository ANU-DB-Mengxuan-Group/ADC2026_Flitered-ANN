#!/bin/bash
# Generate metrics JSONs for validation datasets + build ML validation data.
# Run on cluster (needs access to ~/benchmarks/datasets/discrete/).
#
# Usage:
#   cd ~/benchmarks/discrete
#   conda activate benchmark
#   bash analysis/run_ml_validation.sh
#
# After this completes:
#   git add analysis/ && git commit -m "Add validation metrics + ML data" && git push
#   # Then on local: git pull && python3 analysis/train_ml_router_v2.py --mode both
set -e

cd ~/benchmarks/discrete
DATA_ROOT=~/benchmarks/datasets/discrete

VAL_DATASETS="synth200 arxiv_fanns_real lid50 lid80 lid100 lid120 lid150 synth5 synth30 synth100 hm21"

# ============================================================
# Step 1: Generate metrics JSONs for validation datasets
# ============================================================
echo "========================================="
echo "Step 1: Computing dataset metrics"
echo "========================================="

for DS in $VAL_DATASETS; do
    OUT=analysis/${DS}_metrics.json
    if [ -f "$OUT" ]; then
        echo "  ✅ $DS already exists, skipping"
        continue
    fi
    echo ""
    echo "--- $DS ---"
    python tools/dataset_metrics.py \
        --data-root $DATA_ROOT \
        --dataset $DS \
        --output $OUT \
        --k 100 \
        --query-sample 1000 \
        --base-sample 200000
done

echo ""
echo "Step 1 done. Verifying..."
for DS in $VAL_DATASETS; do
    if [ -f "analysis/${DS}_metrics.json" ]; then
        echo "  ✅ $DS"
    else
        echo "  ❌ $DS MISSING"
    fi
done

# ============================================================
# Step 2: Merge SIEVE into validation recall CSV
# ============================================================
echo ""
echo "========================================="
echo "Step 2: Merging SIEVE into validation recall"
echo "========================================="

MERGED=analysis/ml_router/perquery_recall/perquery_recall_validation_3methods.csv

# Header
head -1 analysis/ml_router/perquery_recall/perquery_recall_validation.csv > $MERGED

# UNG + Post-filter from validation CSV
tail -n +2 analysis/ml_router/perquery_recall/perquery_recall_validation.csv >> $MERGED

# SIEVE from sieve CSV (only validation datasets)
for DS in $VAL_DATASETS; do
    grep "^[0-9].*,$DS," analysis/ml_router/perquery_recall/perquery_recall_sieve.csv >> $MERGED
done

N_LINES=$(wc -l < $MERGED)
echo "  Merged: $N_LINES lines (header + data)"
echo "  Methods: $(tail -n +2 $MERGED | cut -d',' -f4 | sort -u | tr '\n' ' ')"

# ============================================================
# Step 3: Build ML validation data
# ============================================================
echo ""
echo "========================================="
echo "Step 3: Building ML validation data"
echo "========================================="

python analysis/ml_router/build_ml_validation_data.py \
    --data-root $DATA_ROOT \
    --analysis-dir analysis \
    --recall-csv $MERGED \
    --output analysis/ml_validation_data.csv

echo ""
echo "========================================="
echo "All done!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  git add analysis/ && git commit -m 'Add validation metrics + ML data' && git push"
echo "  # On local: git pull"
echo "  # python3 analysis/train_ml_router_v2.py --mode both"
