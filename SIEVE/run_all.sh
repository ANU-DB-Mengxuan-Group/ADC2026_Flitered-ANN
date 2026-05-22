#!/bin/bash
# Run SIEVE experiments on all datasets
# Usage: salloc ... then bash run_all.sh
#
# Features:
# - Skips already converted labels (checks output file existence)
# - Skips already completed experiments (checks CSV has data rows)
# - Can resume after interruption

set -e

DATA_DIR="$HOME/benchmarks/datasets/discrete"
BENCHMARK_DIR="$HOME/benchmarks/discrete"
SIEVE_DIR="$BENCHMARK_DIR/SIEVE"
LABEL_DIR="$SIEVE_DIR/sieve_labels"
RESULT_DIR="$SIEVE_DIR/results"
NUM_THREADS=16

DATASETS=("arxiv" "yfcc" "LAION1M" "tripclick" "ytb_audio" "ytb_video")

# Helper: check if CSV has data (more than just header)
csv_has_data() {
    local csv_file="$1"
    if [ -f "$csv_file" ]; then
        local line_count=$(wc -l < "$csv_file")
        if [ "$line_count" -gt 1 ]; then
            return 0  # has data
        fi
    fi
    return 1  # no data or doesn't exist
}

# -----------------------------------------------------------
# Step 1: Build SIEVE (only if not already installed)
# -----------------------------------------------------------
echo "=== Checking SIEVE installation ==="
cd "$SIEVE_DIR"
if python -c "import hnswlib; print('hnswlib OK')" 2>/dev/null; then
    echo "SIEVE already installed, skipping build"
else
    echo "Building SIEVE..."
    pip install --user pybind11 numpy scipy
    pip install --user .
    echo "SIEVE built successfully"
fi

mkdir -p "$RESULT_DIR"

# -----------------------------------------------------------
# Step 2: Convert labels to SIEVE format (skip if exists)
# -----------------------------------------------------------
echo ""
echo "=== Converting labels (skipping existing) ==="

for ds in "${DATASETS[@]}"; do
    mkdir -p "$LABEL_DIR/$ds"

    # Fixed-EQ labels
    FIXED_EQ_BASE="$LABEL_DIR/$ds/fixed_eq_base_filters.bin"
    FIXED_EQ_QUERY="$LABEL_DIR/$ds/fixed_eq_query_filters.pkl"

    if [ -f "$FIXED_EQ_BASE" ] && [ -f "$FIXED_EQ_QUERY" ]; then
        echo "[$ds] Fixed-EQ labels exist, skipping"
    else
        echo "[$ds] Converting Fixed-EQ labels..."
        # Try ACORN format first (0-based)
        ACORN_BASE="$BENCHMARK_DIR/ACORN/synthetic_labels/$ds/label_base_synthetic.txt"
        ACORN_QUERY="$BENCHMARK_DIR/ACORN/synthetic_labels/$ds/label_query_synthetic.txt"

        if [ -f "$ACORN_BASE" ] && [ -f "$ACORN_QUERY" ]; then
            python "$SIEVE_DIR/convert_labels.py" \
                --base_labels "$ACORN_BASE" \
                --query_labels "$ACORN_QUERY" \
                --output_base "$FIXED_EQ_BASE" \
                --output_query "$FIXED_EQ_QUERY"
        else
            # Fallback to UNG format (1-based)
            UNG_BASE="$BENCHMARK_DIR/UNG-dev/synthetic_labels/$ds/label_base.txt"
            UNG_QUERY="$BENCHMARK_DIR/UNG-dev/synthetic_labels/$ds/label_query.txt"
            if [ -f "$UNG_BASE" ] && [ -f "$UNG_QUERY" ]; then
                echo "  Using UNG labels (1-based -> 0-based)"
                python "$SIEVE_DIR/convert_labels_ung.py" \
                    --base_labels "$UNG_BASE" \
                    --query_labels "$UNG_QUERY" \
                    --output_base "$FIXED_EQ_BASE" \
                    --output_query "$FIXED_EQ_QUERY"
            else
                echo "  WARNING: No Fixed-EQ labels found for $ds"
            fi
        fi
    fi

    # Original labels (1-based, use convert_labels_ung.py)
    ORIG_BASE="$DATA_DIR/$ds/label_base.txt"
    if [ -f "$ORIG_BASE" ]; then
        # AND scenario
        AND_BASE="$LABEL_DIR/$ds/original_and_base_filters.bin"
        AND_QUERY="$LABEL_DIR/$ds/original_and_query_filters.pkl"
        ORIG_QUERY_AND="$DATA_DIR/$ds/${ds}_query_and.txt"

        if [ -f "$AND_BASE" ] && [ -f "$AND_QUERY" ]; then
            echo "[$ds] Original AND labels exist, skipping"
        elif [ -f "$ORIG_QUERY_AND" ]; then
            echo "[$ds] Converting Original AND labels..."
            python "$SIEVE_DIR/convert_labels_ung.py" \
                --base_labels "$ORIG_BASE" \
                --query_labels "$ORIG_QUERY_AND" \
                --output_base "$AND_BASE" \
                --output_query "$AND_QUERY"
        fi

        # OR scenario
        OR_BASE="$LABEL_DIR/$ds/original_or_base_filters.bin"
        OR_QUERY="$LABEL_DIR/$ds/original_or_query_filters.pkl"
        ORIG_QUERY_OR="$DATA_DIR/$ds/${ds}_query_or.txt"

        if [ -f "$OR_BASE" ] && [ -f "$OR_QUERY" ]; then
            echo "[$ds] Original OR labels exist, skipping"
        elif [ -f "$ORIG_QUERY_OR" ]; then
            echo "[$ds] Converting Original OR labels..."
            python "$SIEVE_DIR/convert_labels_ung.py" \
                --base_labels "$ORIG_BASE" \
                --query_labels "$ORIG_QUERY_OR" \
                --output_base "$OR_BASE" \
                --output_query "$OR_QUERY"
        fi
    fi
done

# -----------------------------------------------------------
# Step 3: Run SIEVE experiments (skip completed)
# -----------------------------------------------------------
echo ""
echo "=== Running SIEVE experiments (skipping completed) ==="

# Parameter grid
M_VALUES=(16 32)
BUDGET_VALUES=(1.0 2.0 3.0)
HIST_PCT_VALUES=(0.25 0.50)

total_experiments=0
skipped_experiments=0
completed_experiments=0

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $ds ==="

    for M in "${M_VALUES[@]}"; do
        for budget in "${BUDGET_VALUES[@]}"; do
            for hpct in "${HIST_PCT_VALUES[@]}"; do
                TAG="M${M}_b${budget}_h${hpct}"

                # Fixed-EQ
                if [ -f "$LABEL_DIR/$ds/fixed_eq_base_filters.bin" ]; then
                    total_experiments=$((total_experiments + 1))
                    CSV_FILE="$RESULT_DIR/sieve_${ds}_fixed_eq_${TAG}.csv"

                    if csv_has_data "$CSV_FILE"; then
                        echo "[SKIP] $ds fixed_eq $TAG (already done)"
                        skipped_experiments=$((skipped_experiments + 1))
                    else
                        echo "[RUN] $ds fixed_eq $TAG"
                        python "$SIEVE_DIR/run_sieve.py" \
                            --dataset "$ds" \
                            --scenario fixed_eq \
                            --data_dir "$DATA_DIR" \
                            --label_dir "$LABEL_DIR" \
                            --output_dir "$RESULT_DIR" \
                            --M "$M" \
                            --index_budget "$budget" \
                            --hist_pct "$hpct" \
                            --num_threads "$NUM_THREADS" \
                            2>&1 | tee "$RESULT_DIR/log_${ds}_fixed_eq_${TAG}.txt"
                        completed_experiments=$((completed_experiments + 1))
                    fi
                fi

                # Original AND
                if [ -f "$LABEL_DIR/$ds/original_and_base_filters.bin" ]; then
                    total_experiments=$((total_experiments + 1))
                    CSV_FILE="$RESULT_DIR/sieve_${ds}_original_and_${TAG}.csv"

                    if csv_has_data "$CSV_FILE"; then
                        echo "[SKIP] $ds original_and $TAG (already done)"
                        skipped_experiments=$((skipped_experiments + 1))
                    else
                        echo "[RUN] $ds original_and $TAG"
                        python "$SIEVE_DIR/run_sieve.py" \
                            --dataset "$ds" \
                            --scenario original_and \
                            --data_dir "$DATA_DIR" \
                            --label_dir "$LABEL_DIR" \
                            --output_dir "$RESULT_DIR" \
                            --M "$M" \
                            --index_budget "$budget" \
                            --hist_pct "$hpct" \
                            --num_threads "$NUM_THREADS" \
                            2>&1 | tee "$RESULT_DIR/log_${ds}_original_and_${TAG}.txt"
                        completed_experiments=$((completed_experiments + 1))
                    fi
                fi

                # Original OR
                if [ -f "$LABEL_DIR/$ds/original_or_base_filters.bin" ]; then
                    total_experiments=$((total_experiments + 1))
                    CSV_FILE="$RESULT_DIR/sieve_${ds}_original_or_${TAG}.csv"

                    if csv_has_data "$CSV_FILE"; then
                        echo "[SKIP] $ds original_or $TAG (already done)"
                        skipped_experiments=$((skipped_experiments + 1))
                    else
                        echo "[RUN] $ds original_or $TAG"
                        python "$SIEVE_DIR/run_sieve.py" \
                            --dataset "$ds" \
                            --scenario original_or \
                            --data_dir "$DATA_DIR" \
                            --label_dir "$LABEL_DIR" \
                            --output_dir "$RESULT_DIR" \
                            --M "$M" \
                            --index_budget "$budget" \
                            --hist_pct "$hpct" \
                            --num_threads "$NUM_THREADS" \
                            2>&1 | tee "$RESULT_DIR/log_${ds}_original_or_${TAG}.txt"
                        completed_experiments=$((completed_experiments + 1))
                    fi
                fi
            done
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total experiments: $total_experiments"
echo "Skipped (already done): $skipped_experiments"
echo "Completed this run: $completed_experiments"
echo "Results in: $RESULT_DIR"
