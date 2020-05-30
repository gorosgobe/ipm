#!/usr/bin/env bash

if [[ $# -ne 4 ]]; then
    echo "Expected dataset argument"
    echo "Usage: $0 <dataset> <dsae> <log_file> <seed (random for random seed, give seed otherwise)>"
    exit 1
fi

dataset="$1"
dsae="$2"
log_file="$3"
seed="$4"
source /vol/bitbucket/pg1816/venv/bin/activate

training_list="0.8
"

replication_list="v1
v2
v3
"

for training in $training_list; do
  for repl in $replication_list; do
    echo "Starting training action predictor for DSAE ${dsae}, training data ${training} and replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 train_action_predictor.py --name "act_64_0_1_1_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
            --epochs 200 --dsae_path "$dsae" --latent 128 --training "$training" --seed "$seed" >> "$log_file"
    echo "Completed training."
  done
done

for training in $training_list; do
  for repl in $replication_list; do
    echo "Starting training DSAE chooser for DSAE ${dsae}, training data ${training}, crop size 64, replication ${repl}."
    training_str=$(echo "$training" | sed -e "s/\.//g")
    dataset_str=$(echo "$dataset" | sed -e "s/\///g")
    time python3 choose_feature.py --name "choose_64_0_1_1_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
            --dsae_path "$dsae" --latent 128 --training "$training" --crop_size 64 >> "$log_file"
    echo "Completed training."
    echo "Starting training DSAE chooser for DSAE ${dsae}, training data ${training}, crop size 32, replication ${repl}."
    time python3 choose_feature.py --name "choose_64_0_1_1_32_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
            --dsae_path "$dsae" --latent 128 --training "$training" --crop_size 32 >> "$log_file"
    echo "Completed training."
    echo "Starting training BOP DSAE chooser for DSAE ${dsae}, training data ${training}, replication ${repl}, index obtained from crop search of size 32."
    time python3 choose_feature.py --name "bop_64_0_1_1_32_${dataset_str}_${training_str}_${repl}" --dataset "$dataset" \
            --trials 64 --is_bop yes --index_load_from "choose_64_0_1_1_32_${dataset_str}_${training_str}_${repl}" \
            --dsae_path "$dsae" --latent 128 --training "$training" >> "$log_file"
    echo "Completed training."
  done
done

