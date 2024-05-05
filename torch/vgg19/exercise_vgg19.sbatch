#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH -o exercise_vgg19.out%J
#SBATCH -e exercise_vgg19.err%J

srun \
--container-mounts='../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/torch/vgg19/exercise_vgg19.py
