#!/bin/bash
#SBATCH --job-name=acorn_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --nodelist=weirdo
#SBATCH --output=search_%j.out
#SBATCH --error=search_%j.err

# 进入工作目录
cd ~/benchmarks/discrete/ACORN/bash

# 激活 conda 环境（如果需要）
# source ~/miniconda3/bin/activate benchmark

# 运行参数搜索脚本
python -u auto_param_search_v2.py yfcc LAION1M tripclick ytb_audio ytb_video

echo "任务完成时间: $(date)"
