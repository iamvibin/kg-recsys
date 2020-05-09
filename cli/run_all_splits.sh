#!/bin/bash

declare -a arr=("2" "5")
splits=9
for neighbour in "${arr[@]}"
do
  for split in $(seq 0 $splits);
  do
    ./run.sh ${neighbour} ${split}
  done
done
