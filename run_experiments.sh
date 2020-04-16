#!/bin/bash
for j in 10 15 20 30 40
do
  for i in 1 2 3 4 5
  do
    python train_reward.py --env_name starpilot --models_dir trex/starpilot_model_dir \
    --log_name starpilot"$j" --num_dems "$j" --num_iter 10 --num_snippets 10000 --seed "$i"
  done
done
