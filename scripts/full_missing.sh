#!/usr/bin/env bash

if [[ $# -ne 3 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <log_file> <seed> (random for random seed, give seed otherwise)"
    exit 1
fi

dataset="$1"
dataset_str=$(echo "$dataset" | sed -e "s/\///g")
log_file="$2"
seed="$3"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
0.4
0.2
0.15
0.10
0.05
"

replication_list="v1
v2
v3
"

sizes="64
32
"

versions="coord
"

for training in $training_list; do
  for repl in $replication_list; do
    echo "Starting training full image network with coord for training data ${training}, replication ${repl}"
    training_str=$(echo "$training" | sed -e "s/\.//g")
    time python3 train_tip_velocity_estimator.py --name "FullImageNetwork_${dataset_str}_coord_${training_str}_${repl}" --dataset "$dataset" \
            --training "$training" --seed "$seed" --version "coord" >> "$log_file"
    echo "Completed training"
  done
done

for v in $versions; do
  for s in $sizes; do
    for training in $training_list; do
      for repl in $replication_list; do
        echo "Starting training full image network on size ${s}:${v} data for training data ${training}, replication ${repl}"
        training_str=$(echo "$training" | sed -e "s/\.//g")
        time python3 train_tip_velocity_estimator.py --name "FullImageNetwork_${dataset_str}_${v}_${s}_${training_str}_${repl}" --dataset "$dataset" \
                --training "$training" --size "$s" --version "$v" --seed "$seed" >> "$log_file"
        echo "Completed training."
      done
    done
  done
done

exit 0

