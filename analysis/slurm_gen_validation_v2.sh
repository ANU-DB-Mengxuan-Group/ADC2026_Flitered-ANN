#!/bin/bash
#SBATCH --job-name=gen_validation_v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_validation_v2_%j.out
#SBATCH --error=logs/gen_validation_v2_%j.err

# Generate large-scale synthetic validation datasets (no GPU needed)
# Usage: sbatch analysis/slurm_gen_validation_v2.sh

set -e
cd ~/benchmarks/discrete
mkdir -p logs

echo "=== Generating synthetic validation datasets ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null || ~/anaconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate benchmark

# Generate all 3 synthetic datasets
PYTHONUNBUFFERED=1 python analysis/generate_validation_v2.py --dataset all

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Next steps:"
echo "  1. Generate real-label datasets (yahoo, dbpedia) — needs GPU, see slurm_gen_real_datasets.sh"
echo "  2. Run benchmarks on all datasets"
