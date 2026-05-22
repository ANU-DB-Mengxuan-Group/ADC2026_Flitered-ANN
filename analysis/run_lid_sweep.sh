#!/bin/bash
# Run LID sweep experiments: Post-filter + UNG on lid{50,80,100,120,150}
# Usage: bash analysis/run_lid_sweep.sh
set -e

DATASETS="lid50 lid80 lid100 lid120 lid150"
SCENARIOS="and or equal"
M=32
EFC=100
LBUILD=100
LSEARCH=100
K=10
N=100000
DATA_ROOT=~/benchmarks/datasets/discrete

# =========================================
# Step 1: Convert formats (fvecs->bin, 0-based->1-based, GT->UNG)
# =========================================
echo "Step 1: Converting data formats..."
cd ~/benchmarks/discrete
python analysis/convert_validation_datasets.py

# =========================================
# Step 2: Post-filter
# =========================================
echo ""
echo "========================================="
echo "Step 2: Post-filter experiments"
echo "========================================="

cd ~/benchmarks/discrete/faiss
INDEX_ROOT=~/benchmarks/discrete/faiss/data/index_files/hnsw
RESULT_ROOT=~/benchmarks/discrete/faiss/results_postfilter

for DS in $DATASETS; do
    echo ""
    echo "--- Building Post-filter index for $DS ---"

    for SC in $SCENARIOS; do
        mkdir -p ${RESULT_ROOT}/${DS}/${SC}
    done
    mkdir -p ${INDEX_ROOT}/${DS}

    ./build/tutorial/cpp/build_HNSW_index \
        ${DATA_ROOT}/${DS}/${DS}_base.fvecs \
        $M $EFC \
        $INDEX_ROOT \
        $DS

    for SC in $SCENARIOS; do
        echo "--- Post-filter $DS / $SC ---"
        ./build/tutorial/cpp/search_HNSW_index \
            $DS $M $EFC \
            $INDEX_ROOT \
            $SC \
            ${RESULT_ROOT}/${DS}/${SC} \
            ${DATA_ROOT}/${DS}/${DS}_base.fvecs \
            ${DATA_ROOT}/${DS}/label_base.txt \
            ${DATA_ROOT}/${DS}/${DS}_query_${SC}.fvecs \
            ${DATA_ROOT}/${DS}/${DS}_query_${SC}.txt \
            ${DATA_ROOT}/${DS}/${DS}_gt_${SC}.txt \
            $K $N
    done
done

# =========================================
# Step 3: UNG
# =========================================
echo ""
echo "========================================="
echo "Step 3: UNG experiments"
echo "========================================="

cd ~/benchmarks/discrete/UNG-dev

for DS in $DATASETS; do
    echo ""
    echo "--- Building UNG general index for $DS (AND/OR) ---"
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

    for SCENARIO_MAP in "and:containment" "or:overlap"; do
        SC=${SCENARIO_MAP%%:*}
        UNG_SC=${SCENARIO_MAP##*:}

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

    # Equality needs its own index
    echo "--- Building UNG equality index for $DS ---"
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
echo "LID sweep (Post-filter + UNG) complete!"
echo "SIEVE must run on weirdo node separately: bash analysis/run_lid_sweep_sieve.sh"
