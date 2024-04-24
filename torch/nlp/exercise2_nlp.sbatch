#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:2
#SBATCH -o exercise2.out%J
#SBATCH -e exercise2.err%J

srun \
--container-mounts='./:/workspace' \
--container-image='/dss/dsshome1/0D/di93fuj/containerImages/flash.sqsh' \
python /workspace/exercise2_nlp.py