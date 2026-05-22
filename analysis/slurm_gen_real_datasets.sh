#!/bin/bash
#SBATCH --job-name=gen_real_datasets
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_real_datasets_%j.out
#SBATCH --error=logs/gen_real_datasets_%j.err

# Generate real-label validation datasets (needs GPU for sentence-transformers)
#
# Prerequisites: pip install datasets sentence-transformers torch
#
# Usage: sbatch analysis/slurm_gen_real_datasets.sh

set -e
cd ~/benchmarks/discrete
mkdir -p logs

echo "=== Generating real-label validation datasets ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null || ~/anaconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate benchmark

# --- Yahoo Answers (800K subset, 10 categories, 768d) ---
echo ""
echo "=== Yahoo Answers 800K ==="
PYTHONUNBUFFERED=1 python analysis/prepare_real_dataset.py \
    --dataset yahoo \
    --output_dir ~/benchmarks/datasets/discrete/yahoo800k \
    --name yahoo800k \
    --max_items 800000 \
    --num_queries 1000 \
    --k 10 \
    --batch_size 64

# --- DBpedia-14 (full 560K, 14 categories, 768d) ---
echo ""
echo "=== DBpedia-14 560K ==="
PYTHONUNBUFFERED=1 python analysis/prepare_real_dataset.py \
    --dataset dbpedia14 \
    --output_dir ~/benchmarks/datasets/discrete/dbpedia560k \
    --name dbpedia560k \
    --num_queries 1000 \
    --k 10 \
    --batch_size 64

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Next steps:"
echo "  1. Run benchmarks on all new validation datasets"
