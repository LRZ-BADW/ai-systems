#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH -o exercise1_nlp.out%J
#SBATCH -e exercise1_nlp.err%J
#SBATCH -t 02:00:00

srun \
--container-mounts='./:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/exercise1_nlp.py
