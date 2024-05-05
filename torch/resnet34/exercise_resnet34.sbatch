#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH -o exercise_resnet34.out%J
#SBATCH -e exercise_resnet34.err%J

srun \
--container-mounts='../../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/torch/resnet34/exercise_resnet34.py
