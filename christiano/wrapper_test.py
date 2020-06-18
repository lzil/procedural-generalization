from env_wrapper import gym_procgen_continuous
from procgen import ProcgenGym3Env, ProcgenEnv
from stable_baselines.common import make_vec_env
import numpy as np
import tensorflow as tf
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def no_death_test():
    env = gym_procgen_continuous(env_name = 'fruitbot')

    dones = []
    for i in range(2000): 
        ob, rew, done, info = env.step(0) 
        dones.append(done)

    assert(np.sum(dones) == 0)
    print('Success! : no dones before max_steps timesteps')


def ep_ends_test():
    env = gym_procgen_continuous(env_name = 'fruitbot', max_steps = 1000)

    dones = []
    for i in range(1001): 
        ob, rew, done, info = env.step(0) 
        dones.append(done)

    assert(dones[999])
    assert(sum(dones[:999]) == 0)
    print('Success! : episode ended on max_steps')


def baseline_test():
    from stable_baselines import PPO2
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import env_checker


    gym_env_fn = lambda: gym_procgen_continuous(env_name = 'fruitbot')

    env_checker.check_env(gym_env_fn())
    print('Success! : environment is compatible with stable baselines')

    env = make_vec_env(gym_env_fn, n_envs = 64)

    

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    print('Success! : stable baselines trained on the vectorized environment')


def ProxyRewardWrapper_test():
    from env_wrapper import ProxyRewardWrapper
    env = make_vec_env(lambda: gym_procgen_continuous(env_name = 'fruitbot'), n_envs=4)

    reward_model = lambda x: np.zeros(4)
    new_env = ProxyRewardWrapper(env, reward_model)

    new_env.reset()
    for i in range(11): 
        ob, rew, done, info = new_env.step(4*[new_env.action_space.sample()]) 
        assert((rew == np.zeros(4)).all())
    
    print('Success! : Replaced with zero reward')


if __name__ == "__main__":
    
    no_death_test()
    ep_ends_test()
    baseline_test()
    ProxyRewardWrapper_test()
    