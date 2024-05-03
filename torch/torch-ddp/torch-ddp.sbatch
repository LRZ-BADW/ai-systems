#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:2
#SBATCH -o torch-ddp.out%J
#SBATCH -e torch-ddp.err%J

srun \
--container-mounts='../../:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/containerImages/custom.sqsh' \
python /workspace/torch/torch-ddp/torch-ddp.py