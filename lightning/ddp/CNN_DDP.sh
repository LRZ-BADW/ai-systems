#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:2
#SBATCH -o ./CNN_DDP-%J.out
#SBATCH -t 02:00:00

srun \
--container-mounts='../../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/lightning/ddp/CNN_DDP.py
