#!/bin/bash

for d in /imagenet/val_in_double_folders/*/ ; do
	f="$(basename -- $d)"    
	mkdir $d$f
	mv $d*.JPEG $d$f
done
