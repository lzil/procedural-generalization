import numpy as np
import os
import tensorflow as tf

import argparse

from baselines.ppo2 import ppo2
from procgen import ProcgenEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn


class VideoRunner:
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - make nsteps in the environment using model
    """
    def __init__(self, env_fn, model, nsteps):
        self.env_fn = env_fn
        self.model = model
        self.nsteps = nsteps
        

    def run(self):
        print('Starting to run!')
        venv = self.env_fn()
        self.obs = venv.reset()


        for i in range(self.nsteps):
            if i%100 == 0:
                print(f'Recorded {i} steps')

            #getting actions from the model
            actions, values, self.states, neglogpacs = self.model.step(self.obs)

            # Take actions in env
            self.obs, rewards, self.dones, infos = venv.step(actions[:1])

    
if __name__ == '__main__':
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--video_length', type=int, default=1000, help = 'video length measured in agent steps')
    parser.add_argument('--num_screens', type=int, default=1, help = 'Number of screens in video')
    parser.add_argument('--load_path', type=str, default = 'baseline_agent/sample/checkpoints/coinrun_03000',
        help = 'path to the model')

    args, unknown = parser.parse_known_args()


    #Initializing the model given the environment parameters and path to the saved model
    env_fn = lambda: ProcgenEnv(\
        num_envs=args.num_screens,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode)
    venv_fn  = lambda: VecExtractDictObs(env_fn(), "rgb")

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    model = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0, load_path = args.load_path)

    #this wrapper from openai-baselines records a video
    video_env_fn = lambda: VecVideoRecorder(venv_fn(),
     directory = "Video",
     record_video_trigger=lambda x: x == 1,
     video_length=args.video_length) 

    recorder = VideoRunner(video_env_fn, model, args.video_length)
    recorder.run()


