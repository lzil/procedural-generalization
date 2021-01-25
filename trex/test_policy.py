import os
import tensorflow as tf
import numpy as np
import argparse

from baselines.ppo2 import ppo2
from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn
from helpers.trajectory_collection import ProcgenRunner

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
parser.add_argument('--env_name', type=str, default='chaser')
parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_levels', type=int, default=0)
parser.add_argument('--start_level', type=int, default=0)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--num_runs', type = int, default=128)
parser.add_argument('--load_path', type=str, default = 'trex/reward_model_chaser',
                    help='path to the model')

args, unknown = parser.parse_known_args()

# Initializing the model given the environment parameters and path to the saved model
venv = ProcgenEnv(num_envs=64, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
venv = VecExtractDictObs(venv, "rgb")
conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)
model = ppo2.learn(env=venv, network=conv_fn, total_timesteps=0, load_path=args.load_path)

print('Testing on true reward ...')

venv = lambda: ProcgenEnv(num_envs=64, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
env_fn = lambda: VecExtractDictObs(venv(), "rgb")

collector = ProcgenRunner(env_fn, model, args.max_steps)
eps = collector.collect_episodes(args.num_runs)
print(f'max ep_len = {np.max([ep["length"] for ep in eps])}')
print(f'Mean return ={np.mean([ep["return"] for ep in eps])}')
print('All returns: \n', [ep['return'] for ep in eps])
