#!/bin/bash
for j in 10 20 50 100
do
  for i in 1 2 3 4 5
  do
    python train_reward.py --env_name "$1"  \
    --log_name "$1""$j" --num_dems "$j" --num_iter 5 --num_snippets 20000 --seed "$i"
  done
done
