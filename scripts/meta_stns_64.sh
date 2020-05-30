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
    echo "Starting training normal metastn, scale 0.5, for training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_meta_stn.py --name "final_meta_stn_040401_05_${dataset}_${training}_${repl}" --size 64
    --epochs 200 --split 0.4 0.4 0.1 --loc_lr 0.01 --model_lr 0.1 --batch_size 32 --scale 0.5 >> "$log_file"
    echo "Completed training."

    echo "Starting training DSAE metastn, scale 0.5 for training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_meta_stn.py --name "final_meta_stn_dsae_040401_05_${dataset}_${training}_${repl}" --size 64
    --epochs 200 --split 0.4 0.4 0.1 --loc_lr 0.01 --model_lr 0.1 --batch_size 32 --scale 0.5\
     --dsae_path "target_64_0_1_1_${dataset}_${training}_v1.pt" >> "$log_file"
    echo "Completed training."

    echo "Starting training DSAE metastn with guided init, scale 0.5 for training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_meta_stn.py --name "final_meta_stn_dsae_guided_040401_05_${dataset}_${training}_${repl}" --size 64
    --epochs 200 --split 0.4 0.4 0.1 --loc_lr 0.01 --model_lr 0.1 --batch_size 32 --scale 0.5\
     --dsae_path "target_64_0_1_1_${dataset}_${training}_v1.pt" --dsae_load_index_from "choose_64_0_1_1_${dataset}_${training}_${repl}" >> "$log_file"
    echo "Completed training."

    echo "Starting training DSAE metastn with anneal, scale 0.5 for training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_meta_stn.py --name "final_meta_stn_dsae_anneal_040401_05_${dataset}_${training}_${repl}" --size 64
    --epochs 200 --split 0.4 0.4 0.1 --loc_lr 0.01 --model_lr 0.1 --batch_size 32 --scale 0.5\
     --dsae_path "target_64_0_1_1_${dataset}_${training}_v1.pt" --anneal yes >> "$log_file"
    echo "Completed training."
  done
done