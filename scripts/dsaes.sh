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

replication_list="v1
v2
v3
"

for training in $training_list; do
  for repl in $replication_list; do
    echo "Starting training DSAE autoencoder for training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_dsae.py --name "target_64_0_1_1_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
            --training "$training" --seed "$seed" --g_slow yes --ae_loss_params 0 1 1 --latent 128 --version "target" \
            --batch_size 32 --epochs 300 --output_divisor 4 >> "$log_file"
    echo "Completed training."
  done
done