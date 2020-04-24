import numpy as np
import torch
import pickle
import pandas as pd 
import csv
import os
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

import pdb
import argparse

from helpers.trajectory_collection import generate_procgen_dems

import helpers.baselines_ppo2 as ppo2


parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='starpilot')
parser.add_argument('--distribution_mode', type=str, default='easy',
    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_levels', type=int, default=0)
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--num_dems', default=300, type=int, help="number of trajectories to use")


parser.add_argument('--models_dir', default='trex/experts/0/starpilot/060217/checkpoints')

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with sess.as_default():
    # load environments and generate some number of demonstration trajectories
    procgen_fn_true = lambda: ProcgenEnv(
        num_envs=1,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    venv_fn_true = lambda: VecExtractDictObs(procgen_fn_true(), "rgb")

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    policy_true = ppo2.learn(env=venv_fn_true(), network=conv_fn, total_timesteps=0)


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
        headers = ['path','env_name','mode','length', 'return']
        writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction = 'ignore')

        if not file_exists:
            writer.writeheader()

        num_generated = 0
        while num_generated < args.num_dems:
            dems = generate_procgen_dems(venv_fn_true, policy_true, args.models_dir, max_ep_len=10000, num_dems=10)
            for demo in dems:
                demo['env_name'] = args.env_name
                demo['mode'] = args.distribution_mode
                demo['path'] = os.path.join(demo_dir, str(time.time()) + '.demo')
                pickle.dump(demo, open(demo['path'], 'wb'))

            writer.writerows(dems)
            num_generated += 10
            print(num_generated, ' collected')
            del dems

