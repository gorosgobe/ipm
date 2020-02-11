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

for training in $training_list; do
  echo "Starting training baseline network for training data ${training}."
  training_str=$(echo "$training" | sed -e "s/\.//g")
  python3 train_baseline_estimator.py --name "BaselineNetwork_${dataset}_${training_str}" --dataset "$dataset" \
          --training "$training" >> "$log_file"
  echo "Completed training."
done