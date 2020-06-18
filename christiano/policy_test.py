from train import train_policy, RewardNet
from env_wrapper import gym_procgen_continuous, ProxyRewardWrapper

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy


import tensorflow as tf
import torch
import numpy as np
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env_fn = lambda: gym_procgen_continuous(env_name = 'fruitbot')
policy = PPO2(MlpPolicy, env_fn(), verbose=1)
reward_model = RewardNet()


train_policy(env_fn, reward_model, policy, 1000, device)