#!/bin/bash
LINEARCLASSBUCKET="neurana-imaging"
LINEARCLASSPATH="rhodricusack/deepcluster_analysis/linearclass_v3/"

for ((conv=5;conv>0; conv--)); do
  for ((tp=2;tp<10;tp+=2)); do
      gotfile=$(aws s3 ls "s3://${LINEARCLASSBUCKET}/${LINEARCLASSPATH}linearclass_time_${tp}_conv_${conv}/model_best.pth.tar")
      if [ -z "$gotfile" ]; then
            echo "Time $tp conv $conv not previously run";

            aws sqs send-message --queue-url "https://sqs.eu-west-1.amazonaws.com/807820536621/deepcluster-linearclass.fifo" \
                --message-body "{\"epoch\":$tp,\"conv\":$conv}" \
                --message-group-id "standard5"
      fi
  done
done
         