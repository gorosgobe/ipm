#!/usr/bin/env bash

if [[ $# -ne 3 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <log_file> <seed (random for random seed, give seed otherwise)>"
    exit 1
fi

dataset="$1"
log_file="$2"
seed="$3"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
"

for training in $training_list; do
  echo "Starting training baseline network for training data ${training}."
  training_str=$(echo "$training" | sed -e "s/\.//g")
  dataset_str=$(echo "$dataset" | sed -e "s/\///g")
  time python3 train_baseline_estimator.py --name "BaselineNetwork_${dataset_str}_${training_str}" --dataset "$dataset" \
          --training "$training" --seed "$seed" >> "$log_file"
  echo "Completed training."
done