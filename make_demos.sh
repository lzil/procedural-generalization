
for i in 1 2 3 4 5 
do
	python gen_demos.py  --models_dir trex/experts/0/fruitbot/easy/checkpoints --env_name fruitbot
done

for i in 1 2 
do
	python gen_demos.py  --test_set --models_dir trex/experts/0/fruitbot/easy/checkpoints --env_name fruitbot
done