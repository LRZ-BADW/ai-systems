#!/bin/bash
#SBATCH -N 1
#SBATCH -p test-v100x2 --qos=testing
#SBATCH --gres=gpu:1
#SBATCH -o exercise1_nlp.out%J
#SBATCH -e exercise1_nlp.err%J
#SBATCH -t 02:00:00

srun \
--container-mounts='./:/workspace' \
--container-image='/dss/dsshome1/lxc03/di82hod/pytorch+lightning+custom.sqsh' \
python /workspace/exercise1_nlp.py
