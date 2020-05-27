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

for repl in $replication_list; do
  for training in $training_list; do
    echo "Starting training recurrent full image network, replication ${repl}"
    training_str=$(echo "$training" | sed -e "s/\.//g")
    time python3 train_soft_lstm.py --name "LSTMNetwork_full_${dataset_str}_${training_str}_${repl}" \
    --dataset "$dataset" --batch_size 16 --epochs 300 --training "$training" --version "full" --seed "$seed"\
     --is_coord no --entropy_lambda 0.0 --hidden_size 64 >> "$log_file"
    echo "Completed training."
    echo "Starting training recurrent full coord image network, replication ${repl}"
    time python3 train_soft_lstm.py --name "LSTMNetwork_fullcoord_${dataset_str}_${training_str}_${repl}" \
    --dataset "$dataset" --batch_size 16 --epochs 300 --training "$training" --version "full" --seed "$seed" \
    --is_coord yes --entropy_lambda 0.0 --hidden_size 64 >> "$log_file"
    echo "Completed training."
    echo "Starting training recurrent coordconv32 network, replication ${repl}"
    time python3 train_soft_lstm.py --name "LSTMNetwork_coordconv32_${dataset_str}_${training_str}_${repl}" \
    --dataset "$dataset" --batch_size 16 --epochs 300 --training "$training" --version "coordconv" --seed "$seed"\
    --hidden_size 16 --entropy_lambda 0.0 >> "$log_file"
    echo "Completed training."
    echo "Starting training recurrent mask network, replication ${repl}"
    time python3 train_soft_lstm.py --name "LSTMNetwork_mask_${dataset_str}_${training_str}_${repl}" \
    --dataset "$dataset" --training "$training" --version "soft" --keep_mask yes --hidden_size 85 --seed "$seed"\
     --batch_size 16 --epochs 300 --is_coord yes --entropy_lambda 0.0 >> "$log_file"
    echo "Completed training."
    echo "Starting training recurrent context network, replication ${repl}"
    time python3 train_soft_lstm.py --name "LSTMNetwork_context_${dataset_str}_${training_str}_${repl}" \
    --dataset "$dataset" --batch_size 16 --epochs 300 --training "$training" --version "soft" --seed "$seed"\
     --hidden_size 512 --is_coord yes --entropy_lambda 0.0 >> "$log_file"
    echo "Completed training."
  done
done


exit 0

