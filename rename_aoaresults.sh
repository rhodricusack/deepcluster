#!/bin/bash
LINEARCLASSBUCKET="neurana-imaging"
LINEARCLASSPATH="rhodricusack/deepcluster_analysis/linearclass_v3/"


for ((conv=5;conv>0; conv--)); do
  for ((tp=0;tp<70;tp+=10)); do
#  for ((tp=2;tp<10;tp+=2)); do
      lcpth="s3://${LINEARCLASSBUCKET}/${LINEARCLASSPATH}linearclass_time_${tp}_conv_${conv}"
      aoafile=$(aws s3 ls "$lcpth/aoaresults.json")
      if [ ! -z "$aoafile" ]; then
        echo "Working on $aoafile"
        gotfile=$(aws s3 ls "${lcpth}/model_toplayer" | awk '{print $4}' | tail -n 1 | sed -e s/[^0-9]//g )

        if [ ! -z "$gotfile" ]; then
            echo $gotfile
            aws s3 mv ${lcpth}/aoaresults.json ${lcpth}/aoaresults_toplayer_epoch_$gotfile.json
        fi
      fi
  done
done
          