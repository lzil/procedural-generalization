import numpy as np
import gym


class Gym_procgen_continuous(gym.Wrapper):
  """
  :param env_name: (str) name of the Procgen environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env_name, max_steps=1000, **kwargs):
    kwargs['use_sequential_levels'] = True
    self.env_fn = lambda : gym.make("procgen:procgen-"+ str(env_name) +"-v0", **kwargs) 
    env = self.env_fn()
    # Call the parent constructor, so we can access self.env later
    super(Gym_procgen_continuous, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def set_maxsteps(self, max_steps):
    self.max_steps = max_steps

  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    self.env = self.env_fn()
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    
    # Replace done with negative reward and keep the episode going
    if done:
        reward = -10
        done = False

    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    return obs, np.float(reward), np.bool(done), info


from stable_baselines.common.vec_env import VecEnvWrapper
class Vec_reward_wrapper(VecEnvWrapper):
    """
    This wrapper changes the reward of the provided environment to some function
    of its observations

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
        proxy_rew = self.r_model(obs)
        return obs, proxy_rew, dones, infos

    

class Reward_wrapper(gym.Wrapper):

    def __init__(self, env, r_model):
        super(Reward_wrapper, self).__init__(env)
        assert callable(r_model)
        self.r_model = r_model


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.r_model(observation), done, info
