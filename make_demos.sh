#!/bin/bash

for i in 1 2 3 4 5 
do
	python gen_demos.py  --models_dir trex/experts/fruitbot/easy/checkpoints --env_name fruitbot --name fruitbot_sequential --num_dems 200 --sequential $i
	python gen_demos.py --test_set --models_dir trex/experts/fruitbot/easy/checkpoints --env_name fruitbot --name fruitbot_sequential --sequential $i
done 
