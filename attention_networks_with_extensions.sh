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

versions="coord
tile
"
for v in $versions; do
    for training in $training_list; do
      echo "Starting training attention network for version ${version}, size 64 and training data ${training}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      python3 train_tip_velocity_estimator.py --name "AttentionNetwork${v}${dataset}_${training_str}" --dataset "$dataset" \
              --training "$training" --version "$v" > "$log_file"
      echo "Completed training."
    done
done

for v in $versions; do
    for training in $training_list; do
      echo "Starting training attention network for version ${version}, size 32 and training data ${training}."
      training_str=$(echo "$training" | sed -e "s/\.//g")
      python3 train_tip_velocity_estimator.py --name "AttentionNetwork${v}${dataset}_32_${training_str}" --dataset "$dataset" \
              --training "$training" --version "$v" --size 32 > "$log_file"
      echo "Completed training."
    done
done

exit 0

