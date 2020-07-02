import torch.nn as nn
import torch.optim as optim
import torch

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

from register_policies import ImpalaPolicy
from utils import *

from env_wrapper import Gym_procgen_continuous, Vec_reward_wrapper, Reward_wrapper
import numpy as np
import random
import argparse, pickle

import tensorflow as tf
import os, time, datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips
    
    Each entry is ([clip0, clip1], label)
    clip0, clip2 : lists of observations
    label : float in range [0,1] corresponding to which clip is preferred
    label can also be 0.5 meaning clips are equal, or e.g. 0.95 corresponding 
    to noize in labeling 
    """

    def __init__(self, max_size = 1000):
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

    def val_iter(self):
        'iterator over validation set'
        return iter(self.val_data)

    def get_size(self):
        '''returns buffer size'''
        return self.current_size

    def get_all_pairs(self):
        return np.concatenate((self.train_data, self.val_data))

class RewardNet(nn.Module):
    """Here we set up a callable reward model

    Should have batch normalizatoin and dropout on conv layers
    
    """
    def __init__(self, l2 = 0.01):
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

        self.mean = 0
        self.std = 0.05
        self.l2 = l2

    def forward(self, clip):
        '''
        predicts the sum of rewards of the clip
        '''
        x = clip.permute(0,3,1,2)
        return torch.sum(self.model(x))

    def rew_fn(self, x):
        self.model.eval()
        rewards = torch.squeeze(self.model(x.permute(0,3,1,2))).detach().cpu().numpy()
        rewards = ((rewards / self.std) - self.mean) * 0.05
        return rewards

    def save(self, path):
        torch.save(self.model, path)

    def set_mean_std(self, pairs, device = 'cuda:0'):
        rewards = []
        for clip0, clip1 , label in pairs:
            rewards.extend(self.rew_fn(torch.from_numpy(clip0).float().to(device)))
            rewards.extend(self.rew_fn(torch.from_numpy(clip1).float().to(device)))

        self.mean = np.mean(rewards)
        self.std = np.std(rewards)



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

@timeitt
def calc_val_loss(reward_model, data_buffer, device):

    reward_model.eval()

    loss = 0
    num_pairs = 0
    for clip0, clip1 , label in data_buffer.val_iter():
        ret0 = reward_model(torch.from_numpy(clip0).float().to(device))
        ret1 = reward_model(torch.from_numpy(clip1).float().to(device))
        loss += rm_loss_func(ret0, ret1, label, device).item()
        num_pairs += 1

    av_loss = loss / num_pairs

    return av_loss

@timeitt
def train_reward(reward_model, data_buffer, num_samples, batch_size, device = 'cuda:0'):
    '''
    Traines a given reward_model for num_batches from data_buffer
    Returns new reward_model
    
    Must have:
        Adaptive L2-regularization based on train vs validation loss
        L2-loss on the output
        Output normalized to 0 mean and 0.05 variance across data_buffer
        (Ibarz et al. page 15)
        
    '''
    num_batches = int(num_samples / batch_size)

    reward_model.to(device)
    weight_decay = reward_model.l2
    optimizer = optim.Adam(reward_model.parameters(), lr= 0.0003, weight_decay = weight_decay)
    
    losses = []
    for batch_i in range(num_batches):
        annotations = data_buffer.sample_batch(batch_size)
        loss = 0
        optimizer.zero_grad()
        reward_model.train()
        for clip0, clip1 , label in annotations:
            ret0 = reward_model(torch.from_numpy(clip0).float().to(device))
            ret1 = reward_model(torch.from_numpy(clip1).float().to(device))
            loss += rm_loss_func(ret0, ret1, label, device)
        
        loss = loss / batch_size
        losses.append(loss.item())

        if batch_i % 100 == 0:
            val_loss = calc_val_loss(reward_model, data_buffer, device) 
            av_loss = np.mean(losses[-100:])
            #Adaptive L2 reg
            if val_loss > 1.5 * av_loss:
                for g in optimizer.param_groups: 
                    g['weight_decay'] = g['weight_decay'] * 1.1
                    weight_decay = g['weight_decay']
            elif val_loss < 1.1 * av_loss:
                 for g in optimizer.param_groups:
                    g['weight_decay'] = g['weight_decay'] / 1.1   
                    weight_decay = g['weight_decay']

            print(f'batch : {batch_i}, loss : {av_loss:6.2f}, val loss: {val_loss:6.2f}, L2 : {weight_decay:8.6f}')
            
        loss.backward()
        optimizer.step()

    reward_model.l2 = weight_decay    
    return reward_model

    

@timeitt
def train_policy(venv_fn, reward_model, policy, num_steps, device):
    '''
    Creates new environment by wrapping the env, with Vec_reward_wrapper given the reward_model.
    Traines policy in the new envirionment for num_steps
    Returns retrained policy
    '''

    #creating the environment with reward predicted  from reward_model
    reward_model.to(device)
    proxy_reward_function = lambda x: reward_model.rew_fn(torch.from_numpy(x).float().to(device))
    proxy_reward_venv = Vec_reward_wrapper(venv_fn(), proxy_reward_function)

    policy.set_env(proxy_reward_venv)
    policy.learn(num_steps)

    return policy
    


from collections import namedtuple
Annotation = namedtuple('Annotation', ['clip0', 'clip1', 'label']) 


@timeitt
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
            # skipping clips with same rewards for now
            continue
            #label = 0.5

        data.append(Annotation(np.array(clip0['observations']), np.array(clip1['observations']), label))

    return data


def main():
    ##setup args
    parser = argparse.ArgumentParser(description='Procgen training, with a revised reward model')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--resume_training', action='store_true')

    parser.add_argument('--init_buffer_size', type=int, default=500)
    parser.add_argument('--clip_size', type=int, default=25)
    parser.add_argument('--num_iters', type=int, default=50)
    parser.add_argument('--steps_per_iter', type=int, default=10**6)
    parser.add_argument('--pairs_per_iter', type=int, default=10**5)
    parser.add_argument('--pairs_in_batch', type=int, default=16)
    parser.add_argument('--l2', type=float, default=0.01)


    args = parser.parse_args()

    args.ppo_kwargs = dict(verbose=1, n_steps=256, noptepochs=3, nminibatches = 8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\n Using {device} for training')


    run_dir, monitor_dir, video_dir = setup_logging(args)

    if args.resume_training:
        reward_model, policy, data_buffer, i_num = load_state(run_dir)
        args = load_args(args)

    #initializing objects
    env_fn = lambda: Gym_procgen_continuous(
        env_name = args.env_name, 
        distribution_mode = args.distribution_mode, 
        num_levels = args.num_levels, 
        start_level = args.start_level
        )
    venv_fn  = lambda:  make_vec_env(env_fn, monitor_dir = monitor_dir, n_envs = 32)


    #in case this is a fresh run 
    if not args.resume_training:
        i_num = 0
        policy = PPO2(ImpalaPolicy, venv_fn(), **args.ppo_kwargs)
        reward_model = RewardNet()
        data_buffer = AnnotationBuffer()
        store_args(args, run_dir)   


    

    for i in range(i_num, args.num_iters + i_num):
        print(f'iter : {i}')
        num_pairs = int(args.init_buffer_size / (1+i))
        
        annotations = collect_annotations(env_fn, policy, num_pairs, args.clip_size)
        data_buffer.add(annotations)
        print(f'Buffer size = {data_buffer.get_size()}')
        
        reward_model = train_reward(reward_model, data_buffer, args.pairs_per_iter, args.pairs_in_batch) 
        reward_model.set_mean_std(data_buffer.get_all_pairs())
        policy = train_policy(venv_fn, reward_model, policy, args.steps_per_iter, device)


        eval_env = VecVideoRecorder(DummyVecEnv([env_fn]), video_dir ,
                       record_video_trigger=lambda x: x == 0, video_length=1000,
                       name_prefix="on_iter_{}".format(i))

        proxy_reward_function = lambda x: reward_model.rew_fn(torch.from_numpy(x)[None,:].float().to(device))
        proxy_eval_env = Reward_wrapper(env_fn(), proxy_reward_function)

        print(f'True policy preformance = {evaluate_policy(policy, eval_env, n_eval_episodes=1)}') 
        print(f'Proxy policy preformance = {evaluate_policy(policy, proxy_eval_env, n_eval_episodes=1)}') 
          

        save_state(run_dir, i, reward_model, policy, data_buffer)
        os.rename(monitor_dir, monitor_dir + '_' + str(i))        



if __name__ == '__main__':
    main()
