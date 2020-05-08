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

parser.add_argument('--max_num_epochs', default=None)
parser.add_argument('--epoch_size', default=None)
parser.add_argument('--demo_csv_path', default=None)

parser.add_argument('--config', type=str)  

args = parser.parse_args()

for (seed, env_name, mode, num_dems, max_return, sequential, config) in \
product(range(args.num_seeds),args.env_name, args.distribution_mode,
		args.num_dems, args.max_return, args.sequential):

    command = ['python', 'train_reward.py']

    command.append(f'--env_name={env_name}')
    command.append(f'--distribution_mode={mode}')
    command.append(f'--num_dems={num_dems}')
    command.append(f'--max_return={max_return}')
    command.append(f'--sequential={sequential}')

    if args.config is not None > 0:
        command.append(f'--config={config}')

    if args.max_num_epochs is not None:
        command.append(f'--max_num_epochs={args.max_num_epochs}')
    if args.epoch_size is not None:
        command.append(f'--epoch_size={args.epoch_size}')
    # note: i'm using this because i have separate demo folders for each env
    # thus this won't work well if i'm using multiple environments
    if args.demo_csv_path is not None:
        command.append(f'--demo_csv_path={args.demo_csv_path}')

    command = ' '.join(command)

    print(f'Running: {command}')

    subprocess.call(command, shell=True)
