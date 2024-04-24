#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:1
#SBATCH -o exercise_resnet34.out%J
#SBATCH -e exercise_resnet34.err%J

srun \
--container-mounts='../:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/containerImages/flash.sqsh' \
python /workspace/resnet34/exercise_resnet34.py
