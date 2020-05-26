#!/usr/bin/env bash

if [[ $# -ne 3 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <log_file> <seed(random for random seed, give seed otherwise)>"
    exit 1
fi

dataset="$1"
dataset_str=$(echo "$dataset" | sed -e "s/\///g")
log_file="$2"
seed="$3"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
"

versions="V1
V2
"
for v in $versions; do
    for training in $training_list; do
      echo "Starting training attention network for version ${v}, size 64 and training data ${training}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      time python3 train_attention_tip_velocity_estimator.py --name "AttentionNetwork${v}_${dataset_str}_${training_str}" --dataset "$dataset" \
              --training "$training" --version "$v" --seed "$seed" >> "$log_file"
      echo "Completed training."
    done
done

for v in $versions; do
    for training in $training_list; do
      echo "Starting training attention network for version ${v}, size 32 and training data ${training}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      time python3 train_attention_tip_velocity_estimator.py --name "AttentionNetwork${v}_${dataset_str}_32_${training_str}" --dataset "$dataset" \
              --training "$training" --version "$v" --size 32 --seed "$seed" >> "$log_file"
      echo "Completed training."
    done
done

exit 0

