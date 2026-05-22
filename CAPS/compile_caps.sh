#!/bin/bash
#SBATCH --job-name=caps_compile
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=caps_compile_%j.out
#SBATCH --error=caps_compile_%j.err

source /home/remote/u7905817/miniconda3/etc/profile.d/conda.sh
conda activate benchmark
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

cd /home/remote/u7905817/benchmarks/discrete/CAPS
echo "Compiling CAPS..."
make index 2>&1
make query 2>&1
echo "Compile done!"
ldd ./index | head -10
