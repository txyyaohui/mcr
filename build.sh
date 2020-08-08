#!/bin/bash
bash /etc/profile.d/modules.sh
set -e -x
module load slurm
module load anaconda3 cuda90/toolkit/9.0.176 cudnn/7.0
# one cpu and one gpu
srun --gres=gpu:1 -w node01 python3 k_mnist_xregression_simple.py 

