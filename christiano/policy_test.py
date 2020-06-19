from train import train_policy, RewardNet
from env_wrapper import gym_procgen_continuous, ProxyRewardWrapper

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env

import tensorflow as tf
import torch
import numpy as np
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env_fn = lambda: gym_procgen_continuous(env_name = 'fruitbot', max_steps = 1000)
venv_fn  = lambda:  make_vec_env(env_fn, n_envs = 64)
# It is important to create policy with big n_envs to train fast
policy = PPO2(MlpPolicy, venv_fn(), verbose=1)
reward_model = RewardNet()
#policy.learn(total_timesteps=100000)

train_policy(venv_fn, reward_model, policy, 50000, device)