#!/bin/bash
#SBATCH --job-name=caps_ytb_audio
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=caps_ytb_audio_%j.out
#SBATCH --error=caps_ytb_audio_%j.err

source /home/remote/u7905817/miniconda3/etc/profile.d/conda.sh
conda activate benchmark
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/remote/u7905817/benchmarks/discrete/CAPS

echo "Testing ytb_audio with nb=64..."
rm -rf /home/remote/u7905817/benchmarks/discrete/CAPS/data/indices/ytb_audio/nb_64_slurm
./index /home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_base.fvecs \
        /home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/label_base.txt \
        /home/remote/u7905817/benchmarks/discrete/CAPS/data/indices/ytb_audio/nb_64_slurm \
        64 kmeans 1

echo "Done!"
