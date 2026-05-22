#!/bin/bash
#PBS -N per_label_lid
#PBS -l ncpus=8
#PBS -l mem=64GB
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/z04+gdata/z04
#PBS -l wd
#PBS -j oe
#PBS -o analysis/per_label_lid/job_output.log

# Compute per-label LID for all 13 datasets.
# Usage (interactive): bash analysis/run_per_label_lid.sh
# Usage (PBS):         qsub analysis/run_per_label_lid.sh

cd /home/remote/u7905817/benchmarks/discrete || exit 1

# Activate conda env
source /home/remote/u7905817/miniconda3/etc/profile.d/conda.sh
conda activate benchmark

DATA_ROOT="/home/remote/u7905817/benchmarks/datasets/discrete"
OUTDIR="analysis/per_label_lid"
mkdir -p "$OUTDIR"

# Known global LID values (from *_metrics.json) — avoids recomputation
declare -A GLOBAL_LID
GLOBAL_LID[arxiv]=25.50
GLOBAL_LID[yfcc]=22.99
GLOBAL_LID[LAION1M]=36.31
GLOBAL_LID[tripclick]=31.50
GLOBAL_LID[ytb_audio]=20.49
GLOBAL_LID[ytb_video]=236.04

# ----------- 6 real datasets -----------
for ds in arxiv yfcc LAION1M tripclick ytb_audio ytb_video; do
    echo "=========================================="
    echo "Processing: $ds"
    echo "=========================================="
    python tools/per_label_lid.py \
        --data-root "$DATA_ROOT" \
        --dataset "$ds" \
        --k 100 \
        --min-size 200 \
        --max-labels 500 \
        --query-sample 500 \
        --global-lid "${GLOBAL_LID[$ds]}" \
        --output "${OUTDIR}/${ds}_per_label_lid.json"
    echo ""
done

# ----------- 5 LID sweep datasets -----------
for ds in lid50 lid80 lid100 lid120 lid150; do
    echo "=========================================="
    echo "Processing: $ds"
    echo "=========================================="
    python tools/per_label_lid.py \
        --data-root "$DATA_ROOT" \
        --dataset "$ds" \
        --k 100 \
        --min-size 200 \
        --max-labels 500 \
        --query-sample 500 \
        --output "${OUTDIR}/${ds}_per_label_lid.json"
    echo ""
done

# ----------- 2 validation datasets -----------
for ds in synth200 arxiv_fanns_real; do
    echo "=========================================="
    echo "Processing: $ds"
    echo "=========================================="
    python tools/per_label_lid.py \
        --data-root "$DATA_ROOT" \
        --dataset "$ds" \
        --k 100 \
        --min-size 200 \
        --max-labels 500 \
        --query-sample 500 \
        --output "${OUTDIR}/${ds}_per_label_lid.json"
    echo ""
done

echo "============================================"
echo "All done. Results in ${OUTDIR}/"
echo "============================================"
