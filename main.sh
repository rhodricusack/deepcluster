# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# need to do
#  conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10

DIR="/home/ubuntu/imagenet/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=28
EXP="/home/${USER}/test/exp"
PYTHON="/home/ubuntu/anaconda3/envs/pytorch_p27/bin/python"

mkdir -p ${EXP}

conda activate pytorch_p27

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --checkpoints 5000 --resume /home/ubuntu/test/exp/checkpoints/checkpoint_24.pth.tar
