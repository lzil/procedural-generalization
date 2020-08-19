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

parser.add_argument('--max_num_epochs', default=None)
parser.add_argument('--epoch_size', default=None)
parser.add_argument('--demo_csv', default=None)
parser.add_argument('--demo_folder', default=None)
parser.add_argument('--save_name', default=None)
parser.add_argument('--patience', default=None)

parser.add_argument('--config', default=None, type=str)

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

    if args.config is not None:
        command.append(f'--config={args.config}')

    if args.max_num_epochs is not None:
        command.append(f'--max_num_epochs={args.max_num_epochs}')
    if args.patience is not None:
        command.append(f'--patience={args.patience}')
    if args.epoch_size is not None:
        command.append(f'--epoch_size={args.epoch_size}')
    # note: i'm using this because i have separate demo folders for each env
    # thus this won't work well if i'm using multiple environments
    if args.demo_csv is not None:
        command.append(f'--demo_csv={args.demo_csv}')
    if args.demo_folder is not None:
        command.append(f'--demo_folder={args.demo_folder}')
    if args.save_name is not None:
        command.append(f'--save_name={args.save_name}')

    command = ' '.join(command)

    print(f'Running:\n{command}')

    subprocess.call(command, shell=True)

print(f'Ran {n_exps} experiments.')
