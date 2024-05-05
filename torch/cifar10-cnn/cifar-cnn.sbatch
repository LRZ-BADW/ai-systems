#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH -o cifar-cnn.out%J
#SBATCH -e cifar-cnn.err%J

srun \
--container-mounts='../../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/torch/cifar10-cnn/cifar-cnn.py
