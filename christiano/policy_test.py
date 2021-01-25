from train import train_policy, RewardNet
from env_wrapper import Reward_wrapper, Vec_reward_wrapper, Gym_procgen_continuous

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env
import gym
import tensorflow as tf
import torch
import numpy as np
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

Pay attention to FPS

# Continuous env 
env_fn = lambda: Gym_procgen_continuous(env_name = 'fruitbot', max_steps = 1000)
venv_fn  = lambda:  make_vec_env(env_fn, n_envs = 64)
# It is important to create policy with big n_envs to train fast
policy = PPO2(CnnPolicy, venv_fn(), verbose=1)

policy.learn(total_timesteps=50000)


# Now create the gym env with proxy reward function and then wrap it to make 
# a vectorized environment. FPS is low because reward prediction is not parallelized


reward_model = RewardNet()
reward_model.to(device)
proxy_reward_function = lambda x: reward_model.rew_fn(torch.from_numpy(x)[None,:].float().to(device))
env_fn = lambda: Reward_wrapper(Gym_procgen_continuous(env_name = 'fruitbot', max_steps = 1000), proxy_reward_function)
venv = make_vec_env(env_fn, n_envs = 64)

policy = PPO2(CnnPolicy, venv, verbose=1)
policy.learn(total_timesteps=50000)



#Create a vectorized envirionment, and then replace reward with prediction from reward model
# FPS is high because of parallelization of reward prediciton - but reported statistics is wrong
# Maybe fix it later = Monitor Wrapper provides statictis in the info dictionary

reward_model = RewardNet()
reward_model.to(device)
proxy_reward_function = lambda x: reward_model.rew_fn(torch.from_numpy(x).float().to(device))

env_fn = lambda: Gym_procgen_continuous(env_name = 'fruitbot', max_steps = 100)
venv_fn  = lambda:  make_vec_env(env_fn, n_envs = 64)
venv = Vec_reward_wrapper(venv_fn(), proxy_reward_function)

policy = PPO2(CnnPolicy, venv, verbose=1)
policy.learn(total_timesteps=50000)

# env_name = 'fruitbot'
# env_fn = lambda : gym.make("procgen:procgen-"+ str(env_name) +"-v0", distribution_mode ='easy')
# venv = make_vec_env(env_fn, 64)


# policy = PPO2(CnnPolicy, venv, verbose=1)
# policy.learn(total_timesteps=25 * 10**6)