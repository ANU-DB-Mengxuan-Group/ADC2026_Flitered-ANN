#!/bin/bash
# Run all SIEVE experiments for a single dataset
# Usage: bash run_dataset.sh <dataset>
# Example: bash run_dataset.sh arxiv

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_dataset.sh <dataset>"
    echo "Datasets: arxiv, yfcc, LAION1M, tripclick, ytb_audio, ytb_video"
    exit 1
fi

DATASET="$1"
SIEVE_DIR="$HOME/benchmarks/discrete/SIEVE"
LABEL_DIR="$SIEVE_DIR/sieve_labels"
RESULT_DIR="$SIEVE_DIR/results"

mkdir -p "$RESULT_DIR"

# Parameter grid
M_VALUES=(16 32)
BUDGET_VALUES=(1.0 2.0 3.0)
HIST_PCT_VALUES=(0.25 0.50)
SCENARIOS=()

# Check which scenarios have labels
# Original labels: 3 scenarios (6 datasets)
if [ -f "$LABEL_DIR/$DATASET/original_eq_base_filters.bin" ]; then
    SCENARIOS+=("original_eq")
fi
if [ -f "$LABEL_DIR/$DATASET/original_and_base_filters.bin" ]; then
    SCENARIOS+=("original_and")
fi
if [ -f "$LABEL_DIR/$DATASET/original_or_base_filters.bin" ]; then
    SCENARIOS+=("original_or")
fi
# Fixed-EQ: ACORN synthetic labels (only arxiv & yfcc)
if [ -f "$LABEL_DIR/$DATASET/fixed_eq_base_filters.bin" ]; then
    SCENARIOS+=("fixed_eq")
fi

if [ ${#SCENARIOS[@]} -eq 0 ]; then
    echo "ERROR: No labels found for $DATASET"
    echo "Run 'bash convert_all_labels.sh' first"
    exit 1
fi

echo "=== SIEVE experiments for $DATASET ==="
echo "Scenarios: ${SCENARIOS[*]}"
echo "M values: ${M_VALUES[*]}"
echo "Budget values: ${BUDGET_VALUES[*]}"
echo "Hist_pct values: ${HIST_PCT_VALUES[*]}"
echo ""

# Helper: check if CSV has data (more than just header)
csv_has_data() {
    local csv_file="$1"
    if [ -f "$csv_file" ]; then
        local line_count=$(wc -l < "$csv_file")
        if [ "$line_count" -gt 1 ]; then
            return 0
        fi
    fi
    return 1
}

total=0
skipped=0
completed=0

for scenario in "${SCENARIOS[@]}"; do
    for M in "${M_VALUES[@]}"; do
        for budget in "${BUDGET_VALUES[@]}"; do
            for hpct in "${HIST_PCT_VALUES[@]}"; do
                total=$((total + 1))
                TAG="M${M}_b${budget}_h${hpct}"
                CSV_FILE="$RESULT_DIR/sieve_${DATASET}_${scenario}_${TAG}.csv"

                if csv_has_data "$CSV_FILE"; then
                    echo "[SKIP] $scenario $TAG (already done)"
                    skipped=$((skipped + 1))
                else
                    echo ""
                    echo "[RUN] $DATASET $scenario $TAG"
                    python "$SIEVE_DIR/run_sieve.py" \
                        --dataset "$DATASET" \
                        --scenario "$scenario" \
                        --M "$M" \
                        --index_budget "$budget" \
                        --hist_pct "$hpct" \
                        --validate_gt \
                        2>&1 | tee "$RESULT_DIR/log_${DATASET}_${scenario}_${TAG}.txt"
                    completed=$((completed + 1))
                fi
            done
        done
    done
done

echo ""
echo "=== $DATASET Summary ==="
echo "Total: $total"
echo "Skipped: $skipped"
echo "Completed: $completed"
