#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:2
#SBATCH -o exercise3.out%J
#SBATCH -e exercise3.err%J

srun \
--container-mounts='./:/workspace' \
--container-image='/dss/dsshome1/lxc03/di82hod/pytorch+lightning+custom.sqsh' \
python /workspace/exercise3.py

