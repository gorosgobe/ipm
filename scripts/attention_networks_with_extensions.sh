#!/usr/bin/env bash

if [[ $# -ne 3 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <log_file> <seed (random for random seed, give seed otherwise)>"
    exit 1
fi

dataset="$1"
dataset_str=$(echo "$dataset" | sed -e "s/\///g")
log_file="$2"
seed="$3"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
"

replication_list="v1
v2
v3
"

versions="coord
tile
"
for v in $versions; do
  for repl in $replication_list; do
    for training in $training_list; do
      echo "Starting training attention network for version ${v}, size 64, training data ${training}, replication ${repl}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      time python3 train_attention_tip_velocity_estimator.py --name "AttentionNetwork${v}_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
              --training "$training" --version "$v" --seed "$seed" >> "$log_file"
      echo "Completed training."
    done
  done
done

for v in $versions; do
  for repl in $replication_list; do
    for training in $training_list; do
      echo "Starting training attention network for version ${v}, size 32, training data ${training}, replication ${repl}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      time python3 train_attention_tip_velocity_estimator.py --name "AttentionNetwork${v}_${dataset_str}_32_${training_str}_${repl}" --dataset "$dataset" \
              --training "$training" --version "$v" --size 32 --seed "$seed" >> "$log_file"
      echo "Completed training."
    done
  done
done


exit 0

