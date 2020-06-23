from train import collect_annotations, AnnotationBuffer, RewardNet, train_reward
from env_wrapper import Gym_procgen_continuous
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import torch
import tensorflow as tf
import numpy as np
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



env_fn = lambda: Gym_procgen_continuous(env_name = 'fruitbot')
print('env ready')
policy = PPO2(MlpPolicy, env_fn(), verbose=1)
print('policy ready')

abuffer = AnnotationBuffer(max_size = 10000)

abuffer.add(collect_annotations(env_fn, policy, 20, 25))

sample = abuffer.sample_batch(1)[0]

# print(sample[1])

rm = RewardNet()

print('\nFirst 3 tensors should have different values because of dropout')
print(rm(torch.tensor(sample[0], dtype=torch.float32)))
print(rm(torch.tensor(sample[0], dtype=torch.float32)))
print(rm(torch.tensor(sample[0], dtype=torch.float32)))

print('\nSecond 3 tesors should be same because rm is in eval mode')
rm.eval()
print(rm(torch.tensor(sample[0], dtype=torch.float32)))
print(rm(torch.tensor(sample[0], dtype=torch.float32)))
print(rm(torch.tensor(sample[0], dtype=torch.float32)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import time
t= time.time()
train_reward(rm, abuffer, 100, 8, device)
print(time.time() - t)