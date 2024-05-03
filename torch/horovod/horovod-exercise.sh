#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
##SBATCH -p lrz-v100x2,lrz-hpe-p100x4,lrz-dgx-1-p100x8,lrz-dgx-1-v100x8,lrz-dgx-a100-80x8
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:2
#SBATCH -o horovod-exercise.out%J
#SBATCH -e horovod-exercise.err%J
#SBATCH --container-mounts='../:/workspace'
#SBATCH --container-image='/dss/dsshome1/0D/di93fuj/containerImages/custom.sqsh'

horovodrun -np 2 python /workspace/exercise2.py