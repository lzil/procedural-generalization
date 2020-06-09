from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs

class ProcgenContinuous(ProcgenEnv):

    def __init__(self, max_steps, **kwargs):
        self.max_steps = max_steps
        self.steps = 0
        super().__init__(**kwargs)


    def step(self, actions):
        print('dones')
        ob, rew, dones, infos = super().step(actions)
        self.steps += 1
        
        dones = np.full_like(dones, False)
        if self.steps >= self.max_steps:
            dones = np.full_like(dones, True)

        return ob, rew, dones, infos
