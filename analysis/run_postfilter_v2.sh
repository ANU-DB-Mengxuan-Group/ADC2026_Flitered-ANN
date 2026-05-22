#!/bin/bash
# Run Post-filter HNSW parameter sweep on V2 validation datasets
# Usage: cd ~/benchmarks/discrete && conda activate benchmark && bash analysis/run_postfilter_v2.sh
# Or run single dataset: bash analysis/run_postfilter_v2.sh synth_192d
set -e
cd ~/benchmarks/discrete
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ -n "$1" ]; then
    python faiss/bash/auto_postfilter_hnsw.py "$1"
else
    for DS in synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k; do
        echo ""
        echo "========================================="
        echo "Post-filter: $DS"
        echo "========================================="
        python faiss/bash/auto_postfilter_hnsw.py "$DS"
    done
fi

echo ""
echo "Post-filter V2 validation complete!"
