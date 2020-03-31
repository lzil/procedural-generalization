import os
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
parser.add_argument('--env_name', type=str, default='chaser')
parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--num_levels', type=int, default=1)
parser.add_argument('--start_level', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=500)
parser.add_argument('--load_path', type=str, default = 'trex/reward_model_chaser',
    help = 'path to the model')

args, unknown = parser.parse_known_args()

from baselines.ppo2 import ppo2
from procgen import ProcgenEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env import VecExtractDictObs, VecMonitor
from baselines.common.models import build_impala_cnn

#Initializing the model given the environment parameters and path to the saved model
venv = ProcgenEnv(num_envs=64, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
venv = VecExtractDictObs(venv, "rgb")
conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
model = ppo2.learn(env=venv, network=conv_fn, total_timesteps=0, load_path = args.load_path)

print('Testing on true reward ...')

venv = lambda : ProcgenEnv(num_envs=64, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
env_fn = lambda: VecExtractDictObs(venv(), "rgb")

from helpers.trajectory_collection import ProcgenRunner

collector = ProcgenRunner(env_fn, model, args.max_steps)
eps = collector.collect_episodes(128)
print(f'mean reward ={np.mean([ep["return"] for ep in eps])}')
print('All returns: \n',[ep['return'] for ep in eps])
