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

SQSURL="https://sqs.eu-west-1.amazonaws.com/807820536621/deepcluster-linearclass.fifo"
EXP="${HOME}/linearclass"
echo "${EXP}"
mkdir -p ${EXP}

${PYTHON}  eval_linear_spot.py --data ${DATA} --epochs 2 --lr 0.01 --wd -7 --verbose --exp ${EXP} --workers 8  --checkpointbucket ${CHECKPOINTBUCKET} --checkpointpath ${CHECKPOINTPATH} --sqsurl ${SQSURL}
