#!/bin/bash

for i in 1 
do
	python gen_demos.py  --models_dir trex/experts/bigfish/easy/checkpoints --env_name bigfish --name bigfish_sequential --num_dems 200 --sequential $i
	python gen_demos.py --test_set --models_dir trex/experts/bigfish/easy/checkpoints --env_name bigfish --name bigfish_sequential --sequential $i
done 
