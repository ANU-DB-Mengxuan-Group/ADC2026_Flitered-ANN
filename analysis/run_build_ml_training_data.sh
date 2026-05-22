#!/bin/bash
# Build ML training data (all-in-one, runs on cluster).
#
# Usage (on compute node):
#   bash analysis/run_build_ml_training_data.sh

cd /home/remote/u7905817/benchmarks/discrete || exit 1

source /home/remote/u7905817/miniconda3/etc/profile.d/conda.sh
conda activate benchmark

DATA_ROOT="/home/remote/u7905817/benchmarks/datasets/discrete"

python analysis/ml_router/build_ml_training_data.py \
    --data-root "$DATA_ROOT" \
    --output analysis/ml_training_data.csv

echo ""
echo "Done. Output: analysis/ml_training_data.csv"
echo "Next: git add analysis/ml_training_data.csv && git commit && git push"
