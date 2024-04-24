#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:1
#SBATCH -o exercise_vgg19.out%J
#SBATCH -e exercise_vgg19.err%J

srun \
--container-mounts='../:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/containerImages/flash.sqsh' \
python /workspace/vgg19/exercise_vgg19.py
