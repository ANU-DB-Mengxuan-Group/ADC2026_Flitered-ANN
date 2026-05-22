#!/bin/bash
# Run UNG parameter sweep on V2 validation datasets
# Usage: cd ~/benchmarks/discrete && conda activate ung && bash analysis/run_ung_v2.sh
# Or run single dataset: bash analysis/run_ung_v2.sh synth_192d
set -e
cd ~/benchmarks/discrete
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ -n "$1" ]; then
    python UNG-dev/bash/auto_ung_original.py "$1"
else
    for DS in synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k; do
        echo ""
        echo "========================================="
        echo "UNG: $DS"
        echo "========================================="
        python UNG-dev/bash/auto_ung_original.py "$DS"
    done
fi

echo ""
echo "UNG V2 validation complete!"
