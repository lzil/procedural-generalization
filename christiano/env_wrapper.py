import numpy as np
import gym


class gym_procgen_continuous(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env_name, max_steps=10000, **kwargs):
    kwargs['use_sequential_levels'] = True
    env = gym.make("procgen:procgen-"+ str(env_name) +"-v0", **kwargs) 
    # Call the parent constructor, so we can access self.env later
    super(gym_procgen_continuous, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    
    if done:
        reward = -10
        done = False

    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    return obs, reward, done, info
