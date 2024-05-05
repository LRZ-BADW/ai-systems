#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:2
#SBATCH -o exercise2.out%J
#SBATCH -e exercise2.err%J

srun \
--container-mounts='./:/workspace' \
--container-image='path/to/your/image/image_name.sqsh' \
python /workspace/torch/nlp/exercise2_nlp.py