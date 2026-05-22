#!/bin/bash
#SBATCH --job-name=nhq_test
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodelist=weirdo
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=nhq_%j.log

cd ~/benchmarks/discrete/NHQ

# 测试 arxiv
echo "=== Testing arxiv ==="
python3 test_nhq_synthetic.py arxiv

# 测试 yfcc  
echo "=== Testing yfcc ==="
python3 test_nhq_synthetic.py yfcc

echo "=== Done ==="
