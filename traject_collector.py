import numpy as np
import random


class ProcGenRunner:
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
        self.nsteps = nsteps
        

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
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

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
                episode['observations'] = np.swapaxes(mb_obs,0,1)[i]
                episode['rewards'] = np.swapaxes(mb_rewards,0,1)[i]
                episode['actions'] = np.swapaxes(mb_actions,0,1)[i]
                episode['dones'] = np.swapaxes(mb_dones,0,1)[i]
                batch.append(episode)

        return np.asarray(batch[:n_episodes])


class PreferencesReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.model_index=[]
        self.n_additions = 0
        self.current_size = 0

    def add_episode_pairs(self, runner, n_pairs):
        """Adds a specified number of pairs of episodes to the batch
        stores relevant model index number to self.model_index
        
        episodes are collected using the same model
        """
        episodes = runner.collect_episodes(2 * n_pairs)
        for i in range(n_pairs):
            prefs = self.simple_compare(episodes[2*i], episodes[2*i+1])
            episodes[2*i]['preference'], episodes[2*i+1]['preference'] = prefs

            self.buffer.append(episodes[2*i:2*i+2])

        self.model_index.extend([self.n_additions for _ in range(n_pairs)])
        self.current_size += n_pairs

    def sample(self, batch_size):
        """Returns the batch of trajectories with preferences
        Each entry in a batch has 2 trajectories which a dictionaries
        with keys:
            'observations', 'rewards', 'actions', 'dones', 'preference'
        """
        return random.sample(self.buffer, batch_size)


    def simple_compare(self, traj_1, traj_2):
        """
        Returns the preference as a pair of numbers [0, 1] or [0.5, 0.5]
        """

        if np.sum(traj_1['rewards']) > np.sum(traj_2['rewards']):
            return [1, 0]
        elif np.sum(traj_1['rewards']) < np.sum(traj_2['rewards']):
            return [0, 1]
        else:
            return [0.5, 0.5]

        


