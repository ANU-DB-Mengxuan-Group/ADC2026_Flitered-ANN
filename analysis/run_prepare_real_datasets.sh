#!/bin/bash
# Prepare real-label datasets for filtered ANN benchmarking.
# Run on cluster (needs GPU for embedding generation).
#
# Prerequisites:
#   conda activate benchmark
#   pip install datasets sentence-transformers
#
# Usage:
#   cd ~/benchmarks/discrete
#   bash analysis/run_prepare_real_datasets.sh
set -e

cd ~/benchmarks/discrete
DATA_ROOT=~/benchmarks/datasets/discrete

echo "========================================="
echo "Step 1: DBpedia-14 (100K, 14 categories)"
echo "========================================="
python analysis/prepare_real_dataset.py \
    --dataset dbpedia14 \
    --output_dir $DATA_ROOT/dbpedia14 \
    --name dbpedia14 \
    --max_items 100000 \
    --num_queries 1000

echo ""
echo "========================================="
echo "Step 2: AG News (120K, 4 categories)"
echo "========================================="
python analysis/prepare_real_dataset.py \
    --dataset agnews \
    --output_dir $DATA_ROOT/agnews4 \
    --name agnews4 \
    --num_queries 1000

echo ""
echo "========================================="
echo "Step 3: Yahoo Answers (100K, 10 categories)"
echo "========================================="
python analysis/prepare_real_dataset.py \
    --dataset yahoo \
    --output_dir $DATA_ROOT/yahoo10 \
    --name yahoo10 \
    --max_items 100000 \
    --num_queries 1000

echo ""
echo "========================================="
echo "Done! Next steps:"
echo "========================================="
echo ""
echo "1. Convert formats:"
echo "   python analysis/convert_validation_datasets.py"
echo ""
echo "2. Run benchmarks:"
echo "   bash analysis/run_ung_validation.sh"
echo "   bash analysis/run_postfilter_validation.sh"
echo "   # SIEVE on weirdo node"
echo ""
echo "3. Compute metrics:"
echo "   for ds in dbpedia14 agnews4 yahoo10; do"
echo "     python tools/dataset_metrics.py --data-root \$DATA_ROOT --dataset \$ds"
echo "   done"
