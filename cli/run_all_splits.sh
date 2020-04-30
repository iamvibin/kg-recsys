#!/bin/bash

declare -a arr=("1")
splits=1
for neighbour in "${arr[@]}"
do
  for split in $(seq 0 $splits);
  do
    ./run.sh ${neighbour}_${split}
   # or do whatever with individual element of the array
  done
done
