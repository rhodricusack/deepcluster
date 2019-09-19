#!/bin/bash

for d in /imagenet/val_in_double_folders/*/ ; do
    echo "mv $d $d/"
done