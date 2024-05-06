#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:2
#SBATCH -o torch-ddp.out%J
#SBATCH -e torch-ddp.err%J

srun \
--container-mounts='../../:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/torch/torch-ddp/torch-ddp.py