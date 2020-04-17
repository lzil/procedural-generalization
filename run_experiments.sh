#!/bin/bash
for j in 10 15 20
do
  for i in 1 2 3 4 5
  do
    python train_reward.py --env_name "$1" --models_dir trex/policy_models/"$1" \
    --log_name "$1""$j" --num_dems "$j" --num_iter 10 --num_snippets 10000 --seed "$i"
  done
done
