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

wd_list="V1
V2
V3
V4
"
for wd in $wd_list; do

  for training in $training_list; do
    echo "Starting training full image network for training data ${training} and optim params ${wd}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    time python3 train_tip_velocity_estimator.py --name "FullImageNetworkWD${wd}_${dataset_str}_${training_str}" \
        --dataset "$dataset" --training "$training" --optim "$wd" --seed "$seed" >> "$log_file"
    echo "Completed training."
  done

done

exit 0

