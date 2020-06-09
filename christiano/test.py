from env_wrapper import ProcgenContinuous
from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
import numpy as np

env = ProcgenContinuous(num_envs=8, env_name="coinrun", max_steps = 10000) 
env = VecExtractDictObs(env, "rgb")

env.reset()
dones = []

for i in range(10): 
    ob, rew, done, info = env.step(np.zeros((1,8), dtype = np.int32)) 
    dones.append(done) 
                                