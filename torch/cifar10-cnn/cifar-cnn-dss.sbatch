#!/bin/bash
#SBATCH -p test-v100x2 -q testing
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o cifar-cnn-dss.out%J
#SBATCH -e cifar-cnn-dss.err%J

srun --container-mounts=../../:/workspace,/dss/dssfs04/pn72ka/pn72ka-dss-0000/ajay:/dss_data \
     --container-image='/dss/dsshome1/0D/di93fuj/containerImages/flash.sqsh' \
     python /workspace/torch/cifar10-cnn/cifar-cnn-dss.py
     
     
     