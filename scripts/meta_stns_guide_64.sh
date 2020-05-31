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
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")

    echo "Starting training DSAE metastn with guided init, scale 0.5 for training data ${training_str} and replication ${repl}."
    time python3 train_meta_stn.py --name "final_meta_stn_dsae_guided_040401_05_${dataset_str}_${training_str}_${repl}" --size 64\
    --epochs 200 --dataset "${dataset}" --split 0.4 0.4 0.1 --loc_lr 0.01 --model_lr 0.1 --batch_size 32 --scale 0.5\
     --dsae_path "target_64_0_1_1_${dataset_str}_${training_str}_v1.pt" --dsae_load_index_from "choose_64_0_1_1_${dataset_str}_${training_str}_${repl}" >> "$log_file"
    echo "Completed training."
  done
done