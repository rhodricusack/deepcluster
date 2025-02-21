# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/imagenet"

PYTHON="../anaconda3/bin/python"

CHECKPOINTBUCKET="neurana-imaging"
CHECKPOINTPATH="rhodricusack/iclr2020/deepcluster_analysis/checkpoints_2019-09-11/checkpoints"
LINEARCLASSBUCKET="neurana-imaging"
LINEARCLASSPATH="rhodricusack/iclr2020/deepcluster_analysis/linearclass_v3/"

mkdir -p ${EXP}_${TIMEPOINT}

cd deepcluster
${PYTHON}  eval_linear_spot.py --data ${DATA} --lr 0.01 --wd -7 --verbose --workers 4 --conv ${CONV} --timepoint ${TIMEPOINT}  --checkpointbucket ${CHECKPOINTBUCKET} --checkpointpath ${CHECKPOINTPATH} --linearclassbucket ${LINEARCLASSBUCKET} --linearclasspath ${LINEARCLASSPATH} --aoaval --toplayer_epochs ${TOPLAYER_EPOCHS}

