from train import collect_annotations, AnnotationBuffer
from env_wrapper import gym_procgen_continuous
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

import tensorflow as tf
import numpy as np
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



env_fn = lambda: gym_procgen_continuous(env_name = 'fruitbot')
print('env ready')
policy = PPO2(MlpPolicy, env_fn(), verbose=1)
print('policy ready')

data = collect_annotations(env_fn(), policy, 10, 25)

assert len(data) == 10
print('correct number of pairs collected')


assert(len(data[0].clip0) == len(data[0].clip1) == 25)
print('clips are of correct length')


abuffer = AnnotationBuffer(10000)

abuffer.add(data)
abuffer.add(collect_annotations(env_fn(), policy, 100, 25))
assert abuffer.get_size() == 110
print('Adding data works, buffer size is correct')

assert len(abuffer.sample_batch(5)) == 5
assert len(abuffer.sample_val_batch(5)) == 5
print('Sampling works, sample size is correct')