#!/bin/bash
# Convert all labels to SIEVE format (run once)
# Usage: bash convert_all_labels.sh
#
# Scenarios:
#   - original_eq:  Original labels, Equality queries (6 datasets)
#   - original_and: Original labels, Containment queries (6 datasets)
#   - original_or:  Original labels, Overlap queries (6 datasets)
#   - fixed_eq:     Synthetic labels (ACORN 12-label), Equality queries (arxiv, yfcc only)

set -e

DATA_DIR="$HOME/benchmarks/datasets/discrete"
BENCHMARK_DIR="$HOME/benchmarks/discrete"
SIEVE_DIR="$BENCHMARK_DIR/SIEVE"
LABEL_DIR="$SIEVE_DIR/sieve_labels"

DATASETS=("arxiv" "yfcc" "LAION1M" "tripclick" "ytb_audio" "ytb_video")

echo "=== Converting labels to SIEVE format ==="

for ds in "${DATASETS[@]}"; do
    mkdir -p "$LABEL_DIR/$ds"
    echo ""
    echo "--- $ds ---"

    ORIG_BASE="$DATA_DIR/$ds/label_base.txt"

    # Original-EQ: Original labels + Equality queries
    ORIG_EQ_BASE="$LABEL_DIR/$ds/original_eq_base_filters.bin"
    ORIG_EQ_QUERY="$LABEL_DIR/$ds/original_eq_query_filters.pkl"
    ORIG_QUERY_EQ="$DATA_DIR/$ds/${ds}_query_equal.txt"

    if [ -f "$ORIG_EQ_BASE" ] && [ -f "$ORIG_EQ_QUERY" ]; then
        echo "[SKIP] Original-EQ labels exist"
    elif [ -f "$ORIG_BASE" ] && [ -f "$ORIG_QUERY_EQ" ]; then
        echo "[CONVERT] Original-EQ (1-based -> 0-based)"
        python "$SIEVE_DIR/convert_labels_ung.py" \
            --base_labels "$ORIG_BASE" \
            --query_labels "$ORIG_QUERY_EQ" \
            --output_base "$ORIG_EQ_BASE" \
            --output_query "$ORIG_EQ_QUERY"
    else
        echo "[WARN] No Original-EQ labels for $ds"
    fi

    # Original-AND: Original labels + Containment queries
    ORIG_AND_BASE="$LABEL_DIR/$ds/original_and_base_filters.bin"
    ORIG_AND_QUERY="$LABEL_DIR/$ds/original_and_query_filters.pkl"
    ORIG_QUERY_AND="$DATA_DIR/$ds/${ds}_query_and.txt"

    if [ -f "$ORIG_AND_BASE" ] && [ -f "$ORIG_AND_QUERY" ]; then
        echo "[SKIP] Original-AND labels exist"
    elif [ -f "$ORIG_BASE" ] && [ -f "$ORIG_QUERY_AND" ]; then
        echo "[CONVERT] Original-AND (1-based -> 0-based)"
        python "$SIEVE_DIR/convert_labels_ung.py" \
            --base_labels "$ORIG_BASE" \
            --query_labels "$ORIG_QUERY_AND" \
            --output_base "$ORIG_AND_BASE" \
            --output_query "$ORIG_AND_QUERY"
    else
        echo "[WARN] No Original-AND labels for $ds"
    fi

    # Original-OR: Original labels + Overlap queries
    ORIG_OR_BASE="$LABEL_DIR/$ds/original_or_base_filters.bin"
    ORIG_OR_QUERY="$LABEL_DIR/$ds/original_or_query_filters.pkl"
    ORIG_QUERY_OR="$DATA_DIR/$ds/${ds}_query_or.txt"

    if [ -f "$ORIG_OR_BASE" ] && [ -f "$ORIG_OR_QUERY" ]; then
        echo "[SKIP] Original-OR labels exist"
    elif [ -f "$ORIG_BASE" ] && [ -f "$ORIG_QUERY_OR" ]; then
        echo "[CONVERT] Original-OR (1-based -> 0-based)"
        python "$SIEVE_DIR/convert_labels_ung.py" \
            --base_labels "$ORIG_BASE" \
            --query_labels "$ORIG_QUERY_OR" \
            --output_base "$ORIG_OR_BASE" \
            --output_query "$ORIG_OR_QUERY"
    else
        echo "[WARN] No Original-OR labels for $ds"
    fi

    # Fixed-EQ: ACORN synthetic labels (0-based, 12 labels) - only arxiv & yfcc have GT
    FIXED_EQ_BASE="$LABEL_DIR/$ds/fixed_eq_base_filters.bin"
    FIXED_EQ_QUERY="$LABEL_DIR/$ds/fixed_eq_query_filters.pkl"
    ACORN_BASE="$BENCHMARK_DIR/ACORN/synthetic_labels/$ds/label_base_synthetic.txt"
    ACORN_QUERY="$BENCHMARK_DIR/ACORN/synthetic_labels/$ds/label_query_synthetic.txt"
    ACORN_GT="$BENCHMARK_DIR/ACORN/synthetic_labels/$ds/gt_synthetic.txt"

    if [ -f "$FIXED_EQ_BASE" ] && [ -f "$FIXED_EQ_QUERY" ]; then
        echo "[SKIP] Fixed-EQ labels exist"
    elif [ -f "$ACORN_BASE" ] && [ -f "$ACORN_QUERY" ] && [ -f "$ACORN_GT" ]; then
        echo "[CONVERT] Fixed-EQ from ACORN synthetic (0-based)"
        python "$SIEVE_DIR/convert_labels.py" \
            --base_labels "$ACORN_BASE" \
            --query_labels "$ACORN_QUERY" \
            --output_base "$FIXED_EQ_BASE" \
            --output_query "$FIXED_EQ_QUERY"
    else
        echo "[SKIP] No Fixed-EQ (ACORN synthetic) for $ds"
    fi
done

echo ""
echo "=== Done ==="
echo "Labels saved to: $LABEL_DIR"
