import numpy as np
import torch
import pickle
import pandas as pd 
import csv
import os
import time
import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

import pdb
import argparse

from helpers.trajectory_collection import ProcgenRunner, generate_procgen_dems

import helpers.baselines_ppo2 as ppo2


parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='starpilot')
parser.add_argument('--distribution_mode', type=str, default='easy',
    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--test_set', action = 'store_true')
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--num_dems', default=300, type=int, help="number of trajectories to use")
parser.add_argument('--models_dir', default='trex/experts/0/starpilot/060217/checkpoints')

args = parser.parse_args()


# load environments and generate some number of demonstration trajectories
procgen_fn_true = lambda seed: ProcgenEnv(
    num_envs=1,
    env_name=args.env_name,
    num_levels=1,
    start_level=seed,
    distribution_mode=args.distribution_mode,
)
conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

model_files = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir)]

venv_fn = lambda: VecExtractDictObs(procgen_fn_true(0), "rgb")
init_policy = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0)


demo_dir = 'trex/demos/demo_files'
os.makedirs(demo_dir, exist_ok=True)

# try: 
#     demo_infos = pd.read_csv('trex/demos/demo_infos.csv', index_col=0)
# except:
#     demo_infos = pd.DataFrame(columns =['path','env_name','mode','length', 'return'])

# print(len(demo_infos))
info_path = 'trex/demos/demo_infos.csv'
file_exists = os.path.exists(info_path)

with open (info_path, 'a') as csvfile:
    headers = ['path','env_name','mode','length', 'return','set_name']
    writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction = 'ignore')

    if not file_exists:
        writer.writeheader()

    num_generated = 0
    while num_generated < args.num_dems:
        seed = random.randint(1e8, 1e9 - 1)
        file_name = '_'.join([str(seed)[0:3], str(seed)[3:6], str(seed)[6:9]]) + '.demo'
        if args.test_set:
            set_name = 'TEST'
            seed = int(1e9 + seed)
            file_name = '1_' + file_name
        else:
            set_name = 'TRAIN'
            file_name = '0_' + file_name

        venv_fn = lambda: VecExtractDictObs(procgen_fn_true(seed), "rgb")
        model_path = np.random.choice(model_files)
        init_policy.load(model_path)
        runner = ProcgenRunner(venv_fn, init_policy)

        demo = runner.collect_episodes(1)[0]
        demo['env_name'] = args.env_name
        demo['mode'] = args.distribution_mode
        demo['path'] = os.path.join(demo_dir, file_name)
        demo['set_name'] = set_name

        pickle.dump(demo, open(demo['path'], 'wb'))
        writer.writerow(demo)

        num_generated += 1
        if num_generated % 20 == 0:
            print(num_generated, ' collected')
