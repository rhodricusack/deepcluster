#!/bin/bash

while IFS=$'\t' read -r -a myArray
do
	qsub -v TIMEPOINT=${myArray[0]},CONV=${myArray[1]},TOPLAYER_EPOCHS=2 eval_linear_spot.sh
done < notfound_summarize_performance_v2.csv

