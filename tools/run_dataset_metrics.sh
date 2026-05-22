#!/bin/bash
# Run dataset_metrics.py on all 6 datasets
# Usage: bash tools/run_dataset_metrics.sh

cd /home/remote/u7905817/benchmarks/discrete-filteredANN-benchmark

DATA_ROOT="/home/remote/u7905817/benchmarks/datasets/discrete"

for ds in arxiv yfcc LAION1M tripclick ytb_audio ytb_video; do
    echo "=========================================="
    echo "Processing: $ds"
    echo "=========================================="
    python tools/dataset_metrics.py \
        --data-root "$DATA_ROOT" \
        --dataset "$ds" \
        --query-sample 1000 \
        --base-sample 200000 \
        --k 100
    echo ""
done

echo "Done. Results saved to ${DATA_ROOT}/*_metrics.json"
