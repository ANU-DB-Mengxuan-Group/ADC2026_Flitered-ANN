#!/bin/bash
# Run SIEVE on validation datasets
# Must run on weirdo node!
# Usage: conda activate benchmark && bash analysis/run_sieve_validation.sh
set -e

cd ~/benchmarks/discrete

DATASETS="synth_192d synth_512d synth_768d_hc yahoo800k"
DATA_ROOT=~/benchmarks/datasets/discrete
LABEL_DIR=~/benchmarks/discrete/SIEVE/sieve_labels

# Step 1: Convert labels to SIEVE format (binary CSR + pickle)
echo "========================================="
echo "Converting labels to SIEVE format"
echo "========================================="

for DS in $DATASETS; do
    mkdir -p ${LABEL_DIR}/${DS}

    for SCENARIO_MAP in "and:original_and" "or:original_or" "equal:original_eq"; do
        SC=${SCENARIO_MAP%%:*}
        SIEVE_SC=${SCENARIO_MAP##*:}

        echo "--- Converting $DS / $SC -> $SIEVE_SC ---"

        python SIEVE/convert_labels.py \
            --base_labels ${DATA_ROOT}/${DS}/label_base.txt \
            --query_labels ${DATA_ROOT}/${DS}/${DS}_query_${SC}.txt \
            --output_base ${LABEL_DIR}/${DS}/${SIEVE_SC}_base_filters.bin \
            --output_query ${LABEL_DIR}/${DS}/${SIEVE_SC}_query_filters.pkl
    done
done

# Step 2: Run SIEVE experiments
echo ""
echo "========================================="
echo "Running SIEVE experiments"
echo "========================================="

for DS in $DATASETS; do
    for SIEVE_SC in original_and original_or original_eq; do
        echo ""
        echo "--- SIEVE $DS / $SIEVE_SC ---"

        python SIEVE/run_sieve.py \
            --dataset $DS \
            --scenario $SIEVE_SC \
            --M 32 \
            --ef_construction 40 \
            --index_budget 2.0 \
            --hist_pct 0.25 \
            --num_threads 16
    done
done

echo ""
echo "SIEVE validation complete!"
