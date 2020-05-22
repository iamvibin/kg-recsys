#!/bin/bash
#run all splits.
declare -a arr=("2")
splits=0
for neighbour in "${arr[@]}"
do
  for split in $(seq 0 $splits);
  do
    ./run.sh ${neighbour} ${split} | tee ${neighbour}_${split}.log
  done
done
