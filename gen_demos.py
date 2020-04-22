import numpy as np
import torch
import pickle
import pandas as pd 
import os
import time


from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

import pdb
import argparse

from train_reward import generate_procgen_dems

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

args.seed = np.random.randint(10000)
print(args.seed)

# load environments and generate some number of demonstration trajectories
procgen_fn_true = lambda: ProcgenEnv(
    num_envs=1,
    env_name=args.env_name,
    num_levels=args.num_levels,
    start_level=args.start_level,
    distribution_mode=args.distribution_mode,
    rand_seed = args.seed
)
venv_fn_true = lambda: VecExtractDictObs(procgen_fn_true(), "rgb")

conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
policy_true = ppo2.learn(env=venv_fn_true(), network=conv_fn, total_timesteps=0, seed=args.seed)


demo_dir = 'trex/demos/' + args.env_name +'_' + args.distribution_mode
os.makedirs(demo_dir, exist_ok=True)

try: 
    demo_infos = pd.read_csv(demo_dir+'_demo_infos.csv', index_col=0)
except:
    demo_infos = pd.DataFrame(columns =['path','length', 'return'])

print(len(demo_infos))

num_generated = 0
while num_generated < args.num_dems:
    dems = generate_procgen_dems(venv_fn_true, policy_true, args.models_dir, max_ep_len=10000, num_dems=20)
    
    for demo in dems:
        demo['path'] = os.path.join(demo_dir, str(time.time()) + '.demo')
        pickle.dump(demo, open(demo['path'], 'wb'))
    
    demo_infos = demo_infos.append(pd.DataFrame(dems, columns =['path','length', 'return']) )
    demo_infos.reset_index(drop = True)
    demo_infos.to_csv(demo_dir+'_demo_infos.csv')
    num_generated += 20
    print(num_generated, ' collected')