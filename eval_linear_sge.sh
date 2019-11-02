# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# ARGUMENTS

DATA="/fsx/rhodricusack/imagenet"
PYTHON="~/python"
MODEL="/fsx/rhodricusack/deepcluster_analysis/checkpoints_2019-09-11/checkpoints/checkpoint_0.pth.tar"
EXP="/fsx/rhodricusack/deepcluster_analysis/linearclass_v3/"

echo "CONV is $CONV"
echo "TIMEPOINT is $TIMEPOINT"

~/anaconda2/bin/conda init
~/anaconda2/bin/python ~/deepcluster/eval_linear_spot.py --data ${DATA} --epochs 2 --lr 0.01 --wd -7 --verbose --exp ${EXP}_${TIMEPOINT} --workers 8  \\
            --model ${MODEL}  --aoaval --toplayer_epoch 5 --conv ${CONV} &  
