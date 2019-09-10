# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/ubuntu/imagenet/train"
MODELROOT="${HOME}/deepcluster_models"
MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/linear_classif"

PYTHON="/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python"


mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 3 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
