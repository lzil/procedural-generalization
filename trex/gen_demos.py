import numpy as np
import pickle
import csv
import os
import tensorflow as tf


from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

import argparse

from helpers.trajectory_collection import ProcgenRunner

from baselines.ppo2 import ppo2

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='starpilot')
parser.add_argument('--distribution_mode', type=str, default='easy',
                    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--test_set', action='store_true')
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--num_dems', default=100, type=int, help="number of demonstrations to use")
parser.add_argument('--max_ep_len', default=1000, type=int, help="Max length of the demo")
parser.add_argument('--models_dir', type=str)
parser.add_argument('--sequential', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='demos')
parser.add_argument('--name', type=str, default='NONAME', help="naming for this batch of generated trajectories")


args = parser.parse_args()


# load environments and generate some number of demonstration trajectories
procgen_fn_true = lambda seed: ProcgenEnv(
    num_envs=1,
    env_name=args.env_name,
    num_levels=1,
    start_level=seed,
    distribution_mode=args.distribution_mode,
    use_sequential_levels = args.sequential
)
conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

# check all the policy models in the folder to pull dems from
model_files = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir)]

venv_fn = lambda: VecExtractDictObs(procgen_fn_true(0), "rgb")
init_policy = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0)

if args.name is not None:
    info_path = f'{args.log_dir}/demo_infos_{args.name}.csv'
    demo_dir = f'{args.log_dir}/demo_files_{args.name}'
else:
    info_path = f'{args.log_dir}/demo_infos.csv'
    demo_dir = f'{args.log_dir}/demo_files'

if args.test_set:
    set_name = 'test'
else:
    set_name = 'train'


os.makedirs(demo_dir, exist_ok=True)
file_exists = os.path.exists(info_path)

with open (info_path, 'a') as csvfile:
    headers = ['demo_id', 'env_name', 'mode', 'length', 'return', 'set_name', 'sequential']
    writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')

    if not file_exists:
        writer.writeheader()

    num_generated = 0
    while num_generated < args.num_dems:
        digits = ''.join([str(x) for x in np.random.randint(10, size=9)])
        seed = int(digits)
        if args.test_set:
            seed += 1_000_000_000
            demo_prefix = '1'
        else:
            demo_prefix = '0'
        demo_id = '_'.join([demo_prefix, digits[0:3], digits[3:6], digits[6:9]])
        file_name = demo_id + '.demo'

        if args.sequential:
            seed = args.sequential

        venv_fn = lambda: VecExtractDictObs(procgen_fn_true(seed), "rgb")
        model_path = np.random.choice(model_files)
        init_policy.load(model_path)
        runner = ProcgenRunner(venv_fn, init_policy, nsteps = args.max_ep_len)

        demo = runner.collect_episodes(1)[0]
        demo['env_name'] = args.env_name
        demo['mode'] = args.distribution_mode
        demo['demo_id'] = demo_id
        demo['set_name'] = set_name
        demo['sequential'] = args.sequential

        pickle.dump(demo, open(os.path.join(demo_dir, file_name), 'wb'))
        writer.writerow(demo)

        num_generated += 1
        if num_generated % 20 == 0 or num_generated == args.num_dems:
            print(f'{num_generated}/{args.num_dems} demos collected')
