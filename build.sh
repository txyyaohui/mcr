#!/bin/bash
bash /etc/profile.d/modules.sh
module load slurm
module load anaconda3 cuda90/toolkit/9.0.176 cudnn/7.0
set -e -x
# one cpu and one gpu
srun --gres=gpu:1 -w node01 python3 k_mnist_xregression_simple.py 

