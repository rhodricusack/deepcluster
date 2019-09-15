# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/imagenet"

PYTHON="/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python"

CHECKPOINTBUCKET="neurana-imaging"
CHECKPOINTPATH="rhodricusack/deepcluster_analysis/checkpoints_2019-09-11/checkpoints"
LINEARCLASSBUCKET="neurana-imaging"
LINEARCLASSPATH="rhodricusack/deepcluster_analysis/"

SQSURL="https://sqs.eu-west-1.amazonaws.com/807820536621/deepcluster-linearclass.fifo"
EXP="${HOME}/linearclass"
echo "${EXP}"
mkdir -p ${EXP}

# This will only work for NVIDIA GPUs
NGPUS=`lspci|grep 'NVIDIA'| wc -l`

for ((tp=0;tp<NGPUS;tp++)); do 
    (CUDA_VISIBLE_DEVICES=${tp}; ${PYTHON}  eval_linear_spot.py --data ${DATA} --epochs 2 --lr 0.01 --wd -7 --verbose --exp ${EXP}_${tp} --workers 8  --checkpointbucket ${CHECKPOINTBUCKET} --checkpointpath ${CHECKPOINTPATH} --sqsurl ${SQSURL} --linearclassbucket ${LINEARCLASSBUCKET} --linearclasspath ${LINEARCLASSPATH}) &
done