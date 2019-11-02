# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# ARGUMENTS

while [ $# -gt 0 ]; do
  case "$1" in
    --TIMEPOINT=*)
      TIMEPOINT="${1#*=}"
      ;;
    --CONV=*)
      CONV="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      echo $1
      printf "***************************\n"
      exit 1
  esac
  shift
done


DATA="/fsx/rhodricusack/imagenet"
PYTHON="~/python"
MODEL="/fsx/rhodricusack/deepcluster_analysis/checkpoints_2019-09-11/checkpoints/checkpoint_0.pth.tar"
EXP="/fsx/rhodricusack/deepcluster_analysis/linearclass_v3/"

echo "CONV is $CONV"
echo "TIMEPOINT is $TIMEPOINT"

~/anaconda2/bin/conda init
python eval_linear_spot.py --data ${DATA} --epochs 2 --lr 0.01 --wd -7 --verbose --exp ${EXP}_${TIMEPOINT} --workers 8  \\
            --model ${MODEL}  --aoaval --toplayer_epoch 1) --conv ${CONV} &  
