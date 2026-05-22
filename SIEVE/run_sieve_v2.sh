#!/bin/bash
# SIEVE V2 validation datasets runner

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchmark

datasets="synth_512d synth_768d_hc yahoo800k dbpedia560k"
scenarios="original_and original_or original_eq"

for ds in $datasets; do
  for sc in $scenarios; do
    echo "=== $ds $sc ==="
    python run_sieve.py --dataset "$ds" --scenario "$sc" --M 32 --index_budget 2.0 --hist_pct 0.25
  done
done

echo "All done!"
