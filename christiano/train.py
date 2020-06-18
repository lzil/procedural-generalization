import torch.nn as nn
import torch.optim as optim
import torch

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from env_wrapper import gym_procgen_continuous, ProxyRewardWrapper
import numpy as np
import random

class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips
    
    Each entry is ([clip0, clip1], label)
    clip0, clip2 : lists of observations
    label : float in range [0,1] corresponding to which clip is preferred
    label can also be 0.5 meaning clips are equal, or e.g. 0.95 corresponding 
    to noize in labeling 
    """

    def __init__(self, max_size):
        self.current_size = 0
        self.train_data = []
        self.val_data = []


    def add(self, data):
        '''
        1/e of data goes to the validatation set
        the rest goes to the training set
        '''
        new_val_data , new_train_data = np.split(data, [int(len(data) / np.exp(1))])
        self.val_data.extend(new_val_data)
        self.train_data.extend(new_train_data)
        
        self.current_size += len(data)


    def sample_batch(self, n):
        return random.sample(self.train_data, n)

    def sample_val_batch(self, n):
        return random.sample(self.val_data, n)

    def get_size(self):
        '''returns buffer size'''
        return self.current_size

class RewardNet(nn.Module):
    """Here we set up a callable reward model

    Should have batch normalizatoin and dropout on conv layers
    
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            #conv1
            nn.Dropout2d(p=0.2),
            nn.Conv2d(3, 16, 3, stride=1),
            nn.MaxPool2d(4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            #conv2
            nn.Dropout2d(p=0.2),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.MaxPool2d(4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            #conv3
            nn.Dropout2d(p=0.2),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            # 2 layer mlp
            nn.Flatten(),
            nn.Linear(11*11*16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, clip):
        '''
        predicts the sum of rewards of the clip
        '''
        self.model.train()
        x = clip.permute(0,3,1,2)
        return torch.sum(self.model(x))

    def rew_fn(self, x):
        self.model.eval()
        return torch.squeeze(self.model(x.permute(0,3,1,2))).detach().cpu().numpy()



def rm_loss_func(ret0, ret1, label, device = 'cuda:0'):
    '''custom loss function, to allow for float labels
    unlike nn.CrossEntropyLoss'''

    #compute log(p1), log(p2) where p_i = exp(ret_i) / (exp(ret_1) + exp(ret_2))
    lsm = nn.LogSoftmax(dim = 0)
    log_preds = lsm(torch.stack((ret0, ret1)))

    #compute cross entropy given the label
    target = torch.tensor([1-label, label]).to(device)
    loss = - torch.sum(log_preds * target)

    return loss

def train_reward(reward_model, data_buffer, num_batches, batch_size, device = 'cuda:0'):
    '''
    Traines a given reward_model for num_batches from data_buffer
    Returns new reward_model
    
    Must have:
        Adaptive L2-regularization based on train vs validation loss
        L2-loss on the output
        Output normalized to 0 mean and 0.05 variance across data_buffer
        (Ibarz et al. page 15)
        
    '''

    reward_model.to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr= 0.0003, weight_decay = 0.0001)
    
    #TODO Adaptive L2 reg
    # for g in optimizer.param_groups: 
    #     g['weight_decay'] = g['weight_decay'] * 1.0001

    for batch_i in range(num_batches):

        annotations = data_buffer.sample_batch(batch_size)
        loss = 0
        optimizer.zero_grad()

        for clip0, clip1 , label in annotations:
            ret0 = reward_model(torch.from_numpy(clip0).float().to(device))
            ret1 = reward_model(torch.from_numpy(clip1).float().to(device))
            loss += rm_loss_func(ret0, ret1, label, device)
        
        loss = loss / batch_size
        print(f'batch : {batch_i}, loss : {loss.item():6.2f}')
        loss.backward()
        optimizer.step()

    return reward_model


    


def train_policy(env_fn, reward_model, policy, num_steps, device):
    '''
    Creates new environment by wrapping the env, with ProxyRewardWrapper given the reward_model.
    Traines policy in the new envirionment for num_steps
    Returns retrained policy
    '''

    #creating the environment with reward predicted  from reward_model
    reward_model.to(device)
    proxy_reward_function = lambda x: reward_model.rew_fn(torch.tensor(x).float().to(device))
    vec_env = make_vec_env(env_fn, n_envs = 64)
    proxy_reward_env = ProxyRewardWrapper(vec_env, proxy_reward_function)

    policy.set_env(proxy_reward_env)
    policy.learn(num_steps)

    


from collections import namedtuple
Annotation = namedtuple('Annotation', ['clip0', 'clip1', 'label']) 

def collect_annotations(env_fn, policy, num_pairs, clip_size):
    '''Collects episodes using the provided policy, slices them to snippets of given length,
    selects pairs randomly and annotates 
    Returns a list of named tuples (clip0, clip1, label), where label is float in [0,1]

    '''
    env = env_fn()
    env.set_maxsteps(clip_size * 2 * num_pairs+10)
    clip_pool = []

    obs = env.reset()
    while len(clip_pool) < num_pairs *2:
        clip = {}
        clip['observations'] = []
        clip['return'] = 0
        while len(clip['observations']) < clip_size:
            # _states are only useful when using LSTM policies
            action, _states = policy.predict(obs)
            clip['observations'].append(obs)
            obs, reward, done, info = env.step(action)    
            clip['return'] += reward
            # TODO
            #probably should add noize to observations as a regularization (Ibarz et al. page 15)
        clip_pool.append(clip)

    clip_pairs = np.random.choice(clip_pool, (num_pairs, 2), replace = False)
    data = []
    for clip0, clip1 in clip_pairs:

        if clip0['return'] > clip1['return']:
            label = 0.0
        elif clip0['return'] < clip1['return']:
            label = 1.0 
        elif clip0['return'] == clip1['return']:
            label = 0.5

        data.append(Annotation(np.array(clip0['observations']), np.array(clip1['observations']), label))

    return data


def main():
    ##setup args
    args.init_buffer_size = 500
    args.clip_size = 25
    args.env_name = 'fruitbot'
    args.steps_per_iter = 10**5
    args.pairs_per_iter = 10**5
    args.pairs__in_batch = 16

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #initializing objects
    env_fn = lambda: gym_procgen_continuous(env_name = args.env_name)
    policy = PPO2(MlpPolicy, env_fn(), verbose=1)
    reward_model = RewardNet()
    data_buffer = AnnotationBuffer()


    initial_data = collect_annotations(env_fn, policy, args.init_buffer_size, args.clip_size)
    data_buffer.add(initial_data)

    num_batches = int(args.pairs_per_iter / args.pairs_in_batch)

    for i in args.num_iters:
        num_pairs = get_num_pairs()
        policy_save_path = 'policy' + str(i)
        rm_save_path = 'rm' + str(i)

        reward_model = train_reward(reward_model, data_buffer, num_batches) 
        policy = train_policy(env_fn, reward_model, policy, args.steps_per_iter)
        annotations = collect_annotations(env_fn, policy, num_pairs, args.clip_size)
        data_buffer.add(annotations)
        
        reward_model.save(rm_save_path)
        policy.save(policy_save_path)
