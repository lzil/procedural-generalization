import numpy as np
class VideoRunner:
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env_fn, model, nsteps):
        self.env_fn = env_fn
        self.model = model
        self.nsteps = nsteps
        

    def run(self):
        print('Starting to run!')
        venv = self.env_fn()
        self.obs = venv.reset()

        # For n in range number of steps
        for i in range(self.nsteps):
            if i%100 == 0:
                print(f'Step {i}')
            tile_shape = (int(self.model.act_model.X.shape[0].value/len(self.obs)), 1,1,1)
            model_obs  = np.tile(self.obs, tile_shape )
            actions, values, self.states, neglogpacs = self.model.step(model_obs)

            # Take actions in env and look the results
            self.obs, rewards, self.dones, infos = venv.step(actions[:1])

    
if __name__ == '__main__':
    import os
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import argparse
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--video_length', type=int, default=400, help = 'video length measured in agent steps')
    parser.add_argument('--num_screens', type=int, default=1, help = 'Number of screens in video')
    parser.add_argument('--load_path', type=str, default = 'baseline_agent/sample/checkpoints/coinrun_03000',
        help = 'path to the model')

    args, unknown = parser.parse_known_args()

    from baselines.ppo2 import ppo2
    from procgen import ProcgenEnv
    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    from baselines.common.vec_env import VecExtractDictObs
    from baselines.common.models import build_impala_cnn

    venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    model = ppo2.learn(env=venv, network=conv_fn, total_timesteps=0, load_path = args.load_path, log_interval=1)

    venv = ProcgenEnv(num_envs=args.num_screens, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    env_fn = lambda: VecVideoRecorder(venv, "Video", record_video_trigger=lambda x: x == 1, video_length=args.video_length) 
    recorder = VideoRunner(env_fn, model, args.video_length)
    recorder.run()


