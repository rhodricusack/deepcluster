#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# need to do
#  conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10

DIR="/imagenet/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=28
EXP="$HOME/checkpoints"
PYTHON="/home/ubuntu/anaconda3/envs/pytorch_p27/bin/python"

mkdir -p ${EXP}

conda activate pytorch_p27

# For SPOT instances 
# Now on s3
CHECKPOINTBUCKET="neurana-imaging"
CHECKPOINTPATH="rhodricusack/deepcluster_analysis/checkpoints_2019-09-11/checkpoints"

# Resume from last checkpoint

# if [ -d "${CHECKPOINTPATH}" ]; then 
#   # Get last checkpoint
#   RESUME=$(cd ${CHECKPOINTPATH} && ls checkpoint_* | sort -n -t _ -k 2 | tail -n 1)
#   RESUME_FALLBACK=$(cd ${CHECKPOINTPATH} && ls checkpoint_* | sort -n -t _ -k 2 | tail -n 2 | head -n 1)
# fi

RESUME=$(aws s3 ls s3://$CHECKPOINTBUCKET/$CHECKPOINTPATH/checkpoint_ | awk '{print $4}' | sort -n -t _ -k 2 | tail -n 1)
RESUME_FALLBACK=$(aws s3 ls s3://$CHECKPOINTBUCKET/$$CHECKPOINTPATH/checkpoint_ | awk '{print $4}' | sort -n -t _ -k 2 | tail -n 2 | head -n 1)

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --checkpoints 5000 --resume ${CHECKPOINTPATH}/$RESUME --resume_fallback ${CHECKPOINTPATH}/$RESUME_FALLBACK --bucket $CHECKPOINTBUCKET
