#!/bin/bash
# Extract per-query recall for UNG + Post-filter on validation datasets.
# Can run on any node (login node OK, only reads existing result files).
#
# Usage:
#   cd ~/benchmarks/discrete
#   conda activate benchmark
#   bash analysis/run_extract_validation_perquery.sh

set -e
cd ~/benchmarks/discrete

echo "Extracting per-query recall for validation datasets (UNG + Post-filter)"
echo "======================================================================="

python analysis/ml_router/extract_perquery_recall.py \
    --validation-only \
    --output analysis/ml_router/perquery_recall/perquery_recall_validation.csv

echo ""
echo "Done. Output: analysis/ml_router/perquery_recall/perquery_recall_validation.csv"
echo "Next: run SIEVE per-query on weirdo node (run_sieve_validation_perquery.sh)"
