#!/bin/bash
#SBATCH -N 1
#SBATCH -p ADD_YOUR_PARTITION_HERE
#SBATCH --gres=gpu:2
#SBATCH -o exercise4.out%J
#SBATCH -e exercise4.err%J

srun \
--container-mounts='./:/workspace' \
--container-image='/path/to/image/image.sqsh' \
python /workspace/exercise4_nlp.py

