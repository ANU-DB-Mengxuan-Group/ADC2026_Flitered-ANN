#!/bin/bash
# Run UNG on validation datasets (synth200, arxiv_fanns_real)
# Usage: conda activate ung && bash analysis/run_ung_validation.sh
set -e

cd ~/benchmarks/discrete/UNG-dev

DATASETS="synth5 synth30 synth100 hm21"
M=32
LBUILD=100
LSEARCH=100
K=10
DATA_ROOT=~/benchmarks/datasets/discrete

for DS in $DATASETS; do
    echo ""
    echo "========================================="
    echo "Building UNG general index for $DS (AND/OR)"
    echo "========================================="

    mkdir -p indices_original/${DS}/general

    ./build/apps/build_UNG_index \
        --data_type float \
        --dist_fn L2 \
        --base_bin_file ${DATA_ROOT}/${DS}/${DS}_base.bin \
        --base_label_file ${DATA_ROOT}/${DS}/label_base_1based.txt \
        --index_path_prefix indices_original/${DS}/general/index_M${M}_L${LBUILD} \
        --scenario general \
        --max_degree $M \
        --Lbuild $LBUILD \
        --num_cross_edges 6 \
        --num_threads 16

    # AND and OR use the general index
    for SCENARIO_MAP in "and:containment" "or:overlap"; do
        SC=${SCENARIO_MAP%%:*}
        UNG_SC=${SCENARIO_MAP##*:}

        echo ""
        echo "--- UNG $DS / $SC ($UNG_SC) ---"

        mkdir -p results_original/${DS}/${SC}

        ./build/apps/search_UNG_index \
            --data_type float \
            --dist_fn L2 \
            --base_bin_file ${DATA_ROOT}/${DS}/${DS}_base.bin \
            --query_bin_file ${DATA_ROOT}/${DS}/${DS}_query_${SC}.bin \
            --base_label_file ${DATA_ROOT}/${DS}/label_base_1based.txt \
            --query_label_file ${DATA_ROOT}/${DS}/${DS}_query_${SC}_1based.txt \
            --gt_file ${DATA_ROOT}/${DS}/${DS}_gt_${SC}_ung.bin \
            --K $K \
            --index_path_prefix indices_original/${DS}/general/index_M${M}_L${LBUILD} \
            --scenario ${UNG_SC} \
            --Lsearch $LSEARCH \
            --num_threads 16 \
            --result_path_prefix results_original/${DS}/${SC}/M${M}_L${LBUILD}_Ls${LSEARCH}
    done

    # Equality needs its own index build
    echo ""
    echo "========================================="
    echo "Building UNG equality index for $DS"
    echo "========================================="

    mkdir -p indices_original/${DS}/equality

    ./build/apps/build_UNG_index \
        --data_type float \
        --dist_fn L2 \
        --base_bin_file ${DATA_ROOT}/${DS}/${DS}_base.bin \
        --base_label_file ${DATA_ROOT}/${DS}/label_base_1based.txt \
        --index_path_prefix indices_original/${DS}/equality/index_M${M}_L${LBUILD} \
        --scenario equality \
        --max_degree $M \
        --Lbuild $LBUILD \
        --num_cross_edges 6 \
        --num_threads 16

    echo ""
    echo "--- UNG $DS / equal (equality) ---"

    mkdir -p results_original/${DS}/equal

    ./build/apps/search_UNG_index \
        --data_type float \
        --dist_fn L2 \
        --base_bin_file ${DATA_ROOT}/${DS}/${DS}_base.bin \
        --query_bin_file ${DATA_ROOT}/${DS}/${DS}_query_equal.bin \
        --base_label_file ${DATA_ROOT}/${DS}/label_base_1based.txt \
        --query_label_file ${DATA_ROOT}/${DS}/${DS}_query_equal_1based.txt \
        --gt_file ${DATA_ROOT}/${DS}/${DS}_gt_equal_ung.bin \
        --K $K \
        --index_path_prefix indices_original/${DS}/equality/index_M${M}_L${LBUILD} \
        --scenario equality \
        --Lsearch $LSEARCH \
        --num_threads 16 \
        --result_path_prefix results_original/${DS}/equal/M${M}_L${LBUILD}_Ls${LSEARCH}_eq
done

echo ""
echo "UNG validation complete!"
