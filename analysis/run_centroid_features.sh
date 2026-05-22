#!/bin/bash
# Compute label centroid distance features for ML routing.
# Run on cluster (needs access to base vectors).
#
# Usage:
#   cd ~/benchmarks/discrete
#   conda activate benchmark
#   bash analysis/run_centroid_features.sh
set -e

cd ~/benchmarks/discrete
DATA_ROOT=~/benchmarks/datasets/discrete

echo "========================================="
echo "Step 1: Training set centroid features"
echo "========================================="
python analysis/ml_router/compute_centroid_features.py \
    --data-root $DATA_ROOT \
    --datasets arxiv yfcc LAION1M tripclick ytb_audio ytb_video \
    --output analysis/centroid_features_train.csv

echo ""
echo "========================================="
echo "Step 2: Validation set centroid features"
echo "========================================="
python analysis/ml_router/compute_centroid_features.py \
    --data-root $DATA_ROOT \
    --datasets synth200 arxiv_fanns_real lid50 lid80 lid100 lid120 lid150 \
               synth5 synth30 synth100 hm21 \
    --output analysis/centroid_features_val.csv

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  git add analysis/centroid_features_*.csv && git commit -m 'Add centroid features' && git push"
echo "  # On local: git pull"
echo "  # python3 analysis/train_ml_router_v2.py --mode both --centroid-features"
