#!/bin/bash
#SBATCH -N 1
##SBATCH --partition=lrz-hpe-p100x4,lrz-dgx-1-p100x8,lrz-dgx-1-v100x8,lrz-dgx-a100-80x8
#SBATCH -p test-v100x2
#SBATCH -q testing
#SBATCH --gres=gpu:1
#SBATCH -o cifar10-cnn-%J.out
#SBATCH -t 02:00:00

srun \
--container-mounts='../../:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/dssContainer/ajay/containerImages/custom.sqsh' \
python /workspace/lightning/cifar10-cnn/cifar10-cnn.py