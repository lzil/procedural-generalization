import subprocess
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='Experiments parameters')

parser.add_argument('--env_name', type=str, nargs='+', default=['starpilot'])
parser.add_argument('--distribution_mode', type=str, nargs='+', default=['easy'],
                    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_dems', type=int, nargs='+', default=[15])
parser.add_argument('--num_seeds', type=int, default=5, help="number of random seed for  each experiment")
parser.add_argument('--max_return', type=float, default=[1.0], nargs='+',
                    help='Maximum return of the provided demonstrations \
                     as a fraction of max available return')
parser.add_argument('--sequential', nargs='+', type=int, default=[0])
parser.add_argument('--weight_decay', type=float, default=[0.005], nargs='+',
                    help='Weight decay used for training')

parser.add_argument('--pass_args', default=None, type=str,
                    help="The specified string in quotes would be passed to the train_reward.py script")

args = parser.parse_args()

n_exps = 0

print('Running experiments')

for (seed, env_name, mode, num_dems, max_return, sequential, weight_decay) in \
    product(range(args.num_seeds), args.env_name, args.distribution_mode,
            args.num_dems, args.max_return, args.sequential, args.weight_decay):

    n_exps += 1

    command = ['python', 'train_reward.py']

    command.append(f'--env_name={env_name}')
    command.append(f'--distribution_mode={mode}')
    command.append(f'--num_dems={num_dems}')
    command.append(f'--max_return={max_return}')
    command.append(f'--sequential={sequential}')
    command.append(f'--weight_decay={weight_decay}')

    command.append(args.pass_args)

    command = ' '.join(command)

    print(f'Running:\n{command}')

    subprocess.call(command, shell=True)

print(f'Ran {n_exps} experiments.')
