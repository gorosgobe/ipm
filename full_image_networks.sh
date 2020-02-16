#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <log_file>"
    exit 1
fi

dataset="$1"
log_file="$2"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
0.4
0.2
0.15
0.10
0.05
"

sizes="64
32
"

versions="a
coord
"

for training in $training_list; do
  echo "Starting training full image network with coord for training data ${training}"
  training_str=$(echo "$training" | sed -e "s/\.//g")
  python3 train_tip_velocity_estimator.py --name "FullImageNetwork_${dataset}_coord_${training_str}" --dataset "$dataset" \
          --training "$training" >> "$log_file"
  echo "Completed training"
done

for training in $training_list; do
  echo "Starting training full image network for training data ${training}."
  training_str=$(echo "$training" | sed -e "s/\.//g")
  python3 train_tip_velocity_estimator.py --name "FullImageNetwork_${dataset}_${training_str}" --dataset "$dataset" \
          --training "$training" >> "$log_file"
  echo "Completed training."
done

for v in $versions; do
  for s in $sizes; do
    for training in $training_list; do
      echo "Starting training full image network on size ${s}:${v} data for training data ${training}"
      training_str=$(echo "$training" | sed -e "s/\.//g")
      python3 train_tip_velocity_estimator.py --name "FullImageNetwork_${dataset}_${v}_${s}_${training_str}" --dataset "$dataset" \
              --training "$training" --size "$s" --version "$v" >> "$log_file"
      echo "Completed training."
    done
  done
done

exit 0

