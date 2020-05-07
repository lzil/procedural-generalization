#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
	python gen_demos.py  --models_dir trex/experts/0/maze/easy/checkpoints --env_name maze 
	python gen_demos.py  --models_dir trex/experts/0/coinrun/easy/checkpoints --env_name coinrun
done 

for i in 1 2 
do
	python gen_demos.py  --test_set --models_dir trex/experts/0/maze/easy/checkpoints --env_name maze
	python gen_demos.py  --test_set --models_dir trex/experts/0/coinrun/easy/checkpoints --env_name coinrun
done
