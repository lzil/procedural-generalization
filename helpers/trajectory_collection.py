import numpy as np
import random


class ProcgenRunner:
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
        self.states = model.initial_state
        self.nsteps = nsteps #maximum length of trajectory

    def run(self):
        venv = self.env_fn()
        self.obs = venv.reset()
        self.nenv = nenv = venv.num_envs if hasattr(venv, 'num_envs') else 1
        self.dones = [False for _ in range(nenv)]
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = venv.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)

        #shape of each is [nsteps, nenv, ... ]
        return mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, epinfos

    def collect_episodes(self, n_episodes):
        """Collects enogh episodes using self.run and returns the batch of episodes of specified size
        """  
        batch = []

        while len(batch) < n_episodes:
            mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, epinfos = self.run()
            for i in range(mb_obs.shape[1]):
                episode = dict()
                ep_dones = np.swapaxes(mb_dones,0,1)[i]
                if np.sum(ep_dones) > 0:
                    ep_len = np.where(ep_dones)[0][0]
                else:
                    ep_len = self.nsteps
                episode['observations'] = np.swapaxes(mb_obs,0,1)[i][:ep_len]
                episode['rewards'] = np.swapaxes(mb_rewards,0,1)[i][:ep_len]
                episode['actions'] = np.swapaxes(mb_actions,0,1)[i][:ep_len]
                episode['length'] = ep_len
                episode['return'] = np.sum(episode['rewards'])
                batch.append(episode)

        return np.asarray(batch[:n_episodes])



        


