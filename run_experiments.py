import subprocess
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='Experiments parameters')

parser.add_argument('--env_name', type=str, nargs = '+', default=['starpilot'])
parser.add_argument('--distribution_mode', type=str, nargs = '+',  default=['easy'],
    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_dems', type=int, nargs = '+', default=[15])
parser.add_argument('--num_seeds', type = int, default=5, help="number of random seed for  each experiment")
parser.add_argument('--max_return',type=float , default = [1.0], nargs = '+', 
                        help = 'Maximum return of the provided demonstrations as a fraction of max available return')
parser.add_argument('--sequential', nargs = '+', type = int, default=[0])  
parser.add_argument('--config', nargs = '+', type=str)  

args = parser.parse_args()

for (seed, env_name, mode, num_dems, max_return, sequential, config) in \
product(range(args.num_seeds),args.env_name, args.distribution_mode,
		args.num_dems, args.max_return, args.sequential, args.config):

	subprocess.call(f"python train_reward.py --env_name={env_name} --distribution_mode={mode} \
					--num_dems={num_dems} --max_return={max_return} --sequential={sequential} --config {config}", shell=True)