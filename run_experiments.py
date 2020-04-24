import subprocess
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='Experiments parameters')

parser.add_argument('--env_name', type=str, nargs = '+', default=['starpilot'])
parser.add_argument('--distribution_mode', type=str, nargs = '+',  default=['hard'],
    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_dems', type=int, nargs = '+', default=[10])
parser.add_argument('--seed', default=[0], nargs = '+', help="random seed for experiments")
parser.add_argument('--max_return',type=float , default = [1.0], nargs = '+', 
                        help = 'Maximum return of the provided demonstrations as a fraction of max available return')
   
args = parser.parse_args()

for (env_name, mode, num_dems, seed, max_return) in \
product(args.env_name, args.distribution_mode,
		args.num_dems, args.seed, args.max_return):

	subprocess.call(f"python train_reward.py --env_name={env_name} --distribution_mode={mode} \
					--num_dems={num_dems} --seed={seed} --max_return={max_return}", shell=True)