#!/bin/bash
# Run SIEVE per-query recall extraction on validation datasets.
# MUST run on weirdo node (SIEVE requires it).
#
# Prerequisites:
#   - SIEVE label files must exist (run run_sieve_validation.sh first if needed)
#   - SIEVE result CSVs must exist (to find best configs)
#
# Usage:
#   ssh weirdo
#   cd ~/benchmarks/discrete
#   conda activate benchmark
#   bash analysis/run_sieve_validation_perquery.sh

set -e
cd ~/benchmarks/discrete

echo "SIEVE per-query recall for validation datasets"
echo "==============================================="

# Check if SIEVE label files exist
LABEL_DIR=~/benchmarks/discrete/SIEVE/sieve_labels
MISSING=0
for DS in synth200 arxiv_fanns_real lid50 lid80 lid100 lid120 lid150; do
    if [ ! -d "${LABEL_DIR}/${DS}" ]; then
        echo "WARNING: SIEVE labels missing for ${DS}"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Some SIEVE label files are missing."
    echo "Run label conversion first? (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        DATA_ROOT=~/benchmarks/datasets/discrete
        for DS in synth200 arxiv_fanns_real lid50 lid80 lid100 lid120 lid150; do
            if [ ! -d "${LABEL_DIR}/${DS}" ]; then
                mkdir -p ${LABEL_DIR}/${DS}
                for SCENARIO_MAP in "and:original_and" "or:original_or" "equal:original_eq"; do
                    SC=${SCENARIO_MAP%%:*}
                    SIEVE_SC=${SCENARIO_MAP##*:}
                    echo "Converting $DS / $SC -> $SIEVE_SC"
                    python SIEVE/convert_labels.py \
                        --base_labels ${DATA_ROOT}/${DS}/label_base.txt \
                        --query_labels ${DATA_ROOT}/${DS}/${DS}_query_${SC}.txt \
                        --output_base ${LABEL_DIR}/${DS}/${SIEVE_SC}_base_filters.bin \
                        --output_query ${LABEL_DIR}/${DS}/${SIEVE_SC}_query_filters.pkl
                done
            fi
        done
    fi
fi

echo ""
echo "Running SIEVE per-query extraction..."
python analysis/run_sieve_perquery.py --validation-only

echo ""
echo "Done. Output: analysis/ml_router/perquery_recall/perquery_recall_sieve.csv (merged)"
echo "Next: git add && git commit && git push"
