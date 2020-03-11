from baselines.common.vec_env import VecEnvWrapper



class PredictRewardWrapper(VecEnvWrapper):
    """
    This wrapper changes the reward of the provided environment to some function
    of it's observations

    r_model must be a callable function that takes batch of obervations
    and returns batch of rewards

    """

    def __init__(self, venv, r_model):
        VecEnvWrapper.__init__(self, venv)
        assert callable(r_model)
        self.r_model = r_model

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return obs, self.r_model(obs), dones, infos

    