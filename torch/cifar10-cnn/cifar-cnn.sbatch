#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:1
#SBATCH -o cifar-cnn.out%J
#SBATCH -e cifar-cnn.err%J

srun \
--container-mounts='../:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/containerImages/flash.sqsh' \
python /workspace/cifar10-cnn/cifar-cnn.py
