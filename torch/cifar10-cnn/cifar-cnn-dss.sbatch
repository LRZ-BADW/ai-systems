#!/bin/bash
#SBATCH --partition=ADD_PARTITION_HERE
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o cifar-cnn-dss.out%J
#SBATCH -e cifar-cnn-dss.err%J

srun --container-mounts=../../:/workspace,/dss/dssfs04/pn72ka/pn72ka-dss-0000/ajay:/dss_data \
--container-image='path/to/your/image/image_name.sqsh' \
     python /workspace/torch/cifar10-cnn/cifar-cnn-dss.py
     
     
     