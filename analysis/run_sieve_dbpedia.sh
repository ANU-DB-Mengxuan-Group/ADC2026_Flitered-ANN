#!/bin/bash
# Convert dbpedia560k + run SIEVE on it
# Usage: cd ~/benchmarks/discrete && conda activate benchmark && bash analysis/run_sieve_dbpedia.sh
set -e

cd ~/benchmarks/discrete
DATA_ROOT=~/benchmarks/datasets/discrete
LABEL_DIR=~/benchmarks/discrete/SIEVE/sieve_labels
DS=dbpedia560k

echo "========================================="
echo "Step 1: Convert formats"
echo "========================================="
python analysis/convert_validation_datasets.py

echo ""
echo "========================================="
echo "Step 2: Convert labels to SIEVE format"
echo "========================================="
mkdir -p ${LABEL_DIR}/${DS}
for SCENARIO_MAP in "and:original_and" "or:original_or" "equal:original_eq"; do
    SC=${SCENARIO_MAP%%:*}
    SIEVE_SC=${SCENARIO_MAP##*:}
    echo "--- Converting $DS / $SC -> $SIEVE_SC ---"
    python SIEVE/convert_labels.py --base_labels ${DATA_ROOT}/${DS}/label_base.txt --query_labels ${DATA_ROOT}/${DS}/${DS}_query_${SC}.txt --output_base ${LABEL_DIR}/${DS}/${SIEVE_SC}_base_filters.bin --output_query ${LABEL_DIR}/${DS}/${SIEVE_SC}_query_filters.pkl
done

echo ""
echo "========================================="
echo "Step 3: Run SIEVE experiments"
echo "========================================="
for SIEVE_SC in original_and original_or original_eq; do
    echo ""
    echo "--- SIEVE $DS / $SIEVE_SC ---"
    python SIEVE/run_sieve.py --dataset $DS --scenario $SIEVE_SC --M 32 --ef_construction 40 --index_budget 2.0 --hist_pct 0.25 --num_threads 16
done

echo ""
echo "SIEVE dbpedia560k complete!"
