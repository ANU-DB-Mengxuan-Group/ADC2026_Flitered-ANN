#!/bin/bash
# Run Post-filter on validation datasets (synth200, arxiv_fanns_real)
# Usage: bash analysis/run_postfilter_validation.sh
set -e

cd ~/benchmarks/discrete/faiss

DATASETS="synth5 synth30 synth100 hm21"
SCENARIOS="and or equal"
M=32
EFC=100
K=10
DATA_ROOT=~/benchmarks/datasets/discrete
INDEX_ROOT=~/benchmarks/discrete/faiss/data/index_files/hnsw
RESULT_ROOT=~/benchmarks/discrete/faiss/results_postfilter

for DS in $DATASETS; do
    echo ""
    echo "========================================="
    echo "Building Post-filter index for $DS"
    echo "========================================="

    # Auto-detect N from fvecs header (first 4 bytes = dim, N = filesize / (4 + dim*4))
    FVECS=${DATA_ROOT}/${DS}/${DS}_base.fvecs
    # Read N from the binary: first uint32 in the .bin file
    N=$(python3 -c "import struct; f=open('${DATA_ROOT}/${DS}/${DS}_base.bin','rb'); print(struct.unpack('I',f.read(4))[0])")
    echo "  N=$N"

    for SC in $SCENARIOS; do
        mkdir -p ${RESULT_ROOT}/${DS}/${SC}
    done

    mkdir -p ${INDEX_ROOT}/${DS}

    ./build/tutorial/cpp/build_HNSW_index \
        ${FVECS} \
        $M $EFC \
        $INDEX_ROOT \
        $DS

    for SC in $SCENARIOS; do
        echo ""
        echo "--- Post-filter $DS / $SC ---"
        ./build/tutorial/cpp/search_HNSW_index \
            $DS $M $EFC \
            $INDEX_ROOT \
            $SC \
            ${RESULT_ROOT}/${DS}/${SC} \
            ${FVECS} \
            ${DATA_ROOT}/${DS}/label_base.txt \
            ${DATA_ROOT}/${DS}/${DS}_query_${SC}.fvecs \
            ${DATA_ROOT}/${DS}/${DS}_query_${SC}.txt \
            ${DATA_ROOT}/${DS}/${DS}_gt_${SC}.txt \
            $K $N
    done
done

echo ""
echo "Post-filter validation complete!"
