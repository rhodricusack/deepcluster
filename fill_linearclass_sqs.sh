#!/bin/bash
LINEARCLASSBUCKET="neurana-imaging"
LINEARCLASSPATH="rhodricusack/deepcluster_analysis/linearclass/"

for ((tp=70;tp>=0;tp-=10)); do
  for ((conv=5;conv>0; conv--)); do
      gotfile=$(aws s3 ls "s3://${LINEARCLASSBUCKET}/${LINEARCLASSPATH}linearclass_v2_time_${tp}_conv_${conv}")
      if [ -z "$gotfile" ]; then
            echo "Nothing found for time $tp conv $conv";

            aws sqs send-message --queue-url "https://sqs.eu-west-1.amazonaws.com/807820536621/deepcluster-linearclass.fifo" \
                --message-body "{\"epoch\":$tp,\"conv\":$conv}" \
                --message-group-id "standard5"
      fi
  done
done
         