#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH -o vgg-resnet-%J.out
#SBATCH -t 02:00:00

srun \
--container-mounts='../../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/lightning/vgg-resnet/vgg-resnet.py