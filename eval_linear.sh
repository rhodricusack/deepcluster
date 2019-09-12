# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

STARTOFFSET=$1
ENDPOINT=$2
STEPSIZE=$3

echo "Startoffset is $1, endpoint $2 and step size $£"

DATA="/home/ubuntu/imagenet"
EXP="${HOME}/deepcluster_exp/linear_classif"

PYTHON="/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python"

for ((tp=STARTOFFSET;tp<=ENDPOINT;tp+=STEPSIZE)); do
  MODEL="${HOME}/test/exp/checkpoints/checkpoint_${tp}.pth.tar"
  for ((conv=5;conv>0; conv--)); do
    EXP="${HOME}/deepcluster_analysis/linearclass_time_${tp}_conv_${conv}"
    echo "Deepcluster tp ${tp} block ${conv}"
    echo "${EXP}"
    mkdir -p ${EXP}
    # Specific GPUs CUDA_VISIBLE_DEVICES=$STARTOFFSET 
    ${PYTHON}  eval_linear.py --model ${MODEL} --data ${DATA} --epochs 2 --conv ${conv} --lr 0.01 --wd -7 --verbose --exp ${EXP} --workers 28 
    # Cannot use tencrops option as the lambda function breaks pickle in multiprocessing.reduction.dump
  done
done