#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:2
#SBATCH -o horovod-exercise.out%J
#SBATCH -e horovod-exercise.err%J
#SBATCH --container-mounts='../../:/workspace'
#SBATCH --container-image='path/to/your/image/image_name.sqsh'

horovodrun -np 2 python /workspace/torch/horovod/horovod-exercise.py