#!/bin/bash

declare -a arr=("1")
splits=1
for neighbour in "${arr[@]}"
do
  for split in $(seq 0 $splits);
  do
    ./run.sh ${neighbour} ${split}
  done
done
