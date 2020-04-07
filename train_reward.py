# adapted heavily from https://github.com/hiwonjoon/ICML2019-TREX/blob/master/atari/LearnAtariReward.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import os
import random

import tensorflow as tf

from baselines.ppo2 import ppo2
from procgen import ProcgenEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

from helpers.trajectory_collection import ProcgenRunner
from helpers.utils import *

import argparse


def generate_procgen_dems(env_fn, model, model_dir, max_ep_len, num_dems):
    
    """
    loop through models in model_dir and sample demonstrations
    until num_dems demonstrations is collected
    """

    dems = []
    while len(dems) < num_dems:
        for model_file in os.scandir(model_dir):
            model.load(model_file)
            collector = ProcgenRunner(env_fn, model, max_ep_len)
            dems.extend(collector.collect_episodes(1)) #collects one episode with current model

    return dems[:num_dems]

def create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length):
    """
    This function takes a set of demonstrations and produces 
    a training set consisting of pairs of clips with assigned preferences
    """

    #Print out some info
    print(len(dems), ' demonstrations provided')
    print("demo lengths :", [d['length'] for d in dems])
    print('demo returns :', [d['return'] for d in dems])
    demo_lens = [d['length'] for d in dems]
    print(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')
    assert min_snippet_length < min(demo_lens), "One of the trajectories is too short"
    
    training_data = []
    
    for n in range(num_snippets):

        #pick two random demos
        two_dems = random.sample(dems, 2)
        # d1['return'] <= d2['return']
        d0, d1 = sorted(two_dems, key = lambda x: x['return'])
        #create random snippets
        
        #first adjust max stippet length such that we can pick
        #the later starting clip from the better trajectory
        cur_min_len = min(d0['length'], d1['length'])
        cur_max_snippet_len = min(cur_min_len, max_snippet_length)
        #randomly choose snipped length
        cur_len = np.random.randint(min_snippet_length, cur_max_snippet_len)

        #pick tj snippet to be later than ti
        d0_start = np.random.randint(cur_min_len - cur_len + 1)
        d1_start = np.random.randint(d0_start, d1['length'] - cur_len + 1)

        clip0  = d0['observations'][d0_start : d0_start+cur_len]
        clip1  = d1['observations'][d1_start : d1_start+cur_len]

        # randomize label so reward learning model won't learn heuristic
        label = np.random.randint(2)
        if label:
            training_data.append(([clip0, clip1], np.array([1])))
        else:
            training_data.append(([clip1, clip0], np.array([0])))

    return np.array(training_data)


# actual reward learning network
class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16*16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def predict_returns(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        r = self.model(x)
        all_reward = torch.sum(r)
        all_reward_abs = torch.sum(torch.abs(r))
        return all_reward, all_reward_abs

    def predict_batch_rewards(self, batch_obs):
        with torch.no_grad():
            x = torch.tensor(batch_obs, dtype=torch.float32).permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
            r = self.model(x)
            return r.numpy().flatten()

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        all_r_i, abs_r_i = self.predict_returns(traj_i)
        all_r_j, abs_r_j = self.predict_returns(traj_j)
        return torch.stack((all_r_i, all_r_j)), abs_r_i + abs_r_j


# trainer wrapper in order to make training the reward model a neat process
class RewardTrainer:
    def __init__(self, args, device):
        self.device = device
        self.net = RewardNet().to(device)

        self.args = args

    # Train the network
    def learn_reward(self, training_data):
        loss_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
                  
        cum_loss = 0.0
        for epoch in range(self.args.num_iter):

            np.random.shuffle(training_data)
            for i, ([traj_i, traj_j], label) in enumerate(training_data):

                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)
                label = torch.from_numpy(label).to(self.device)

                optimizer.zero_grad()

                #forward + backward + optimize
                outputs, abs_rewards = self.net.forward(traj_i, traj_j)
                outputs = outputs.unsqueeze(0)
                # TODO: confirm the dimensionality here is correct. not totally sure
                # TODO: consider l2 regularization?
                loss = loss_criterion(outputs, label) + self.args.lam_l1 * abs_rewards
                loss.backward()
                optimizer.step()

                item_loss = loss.item()
                cum_loss += item_loss
                i+=1
                if i % 1000 == 999:
                    print("epoch {}, step {}: loss {}".format(epoch,i, cum_loss))
                    print(f'absolute rewards = {abs_rewards.item()}')
                    cum_loss = 0.0
                    # TODO: give this a different name for each log so it doesn't keep overwriting
                    torch.save(self.net.state_dict(), self.args.reward_model_path)

            # TODO (max): might want to calculate absolute accuracy every epoch or so

        print("finished training")

    # save the final learned model
    def save_model(self):
        torch.save(self.net.state_dict(), self.args.reward_model_path)

    # calculate and return accuracy on entire training set
    def calc_accuracy(self, training_data):
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            # TODO: use a DataLoader
            for [traj_i, traj_j], label in training_data:
                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)

                #forward to get logits
                outputs, abs_return = self.net.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs,0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_data)


    # purpose of these two functions is to get predicted return (via reward net) from the trajectory given as input
    def predict_reward_sequence(self, traj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rewards_from_obs = []
        with torch.no_grad():
            for s in traj:
                r = self.net.predict_returns(torch.from_numpy(np.array([s])).float().to(device))[0].item()
                rewards_from_obs.append(r)
        return rewards_from_obs

    def predict_traj_return(self, traj):
        return sum(self.predict_reward_sequence(traj))

def parse_config():
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='chaser')
    parser.add_argument('--distribution_mode', type=str, default='hard',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_snippets', default=6000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--models_dir', default = "trex/chaser_model_dir", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--reward_model_path', default='trex/reward_model_chaser', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--num_dems',type=int, default = 12 , help = 'Number of demonstrations to train on')
    args = parser.parse_args()

    # TODO (max): change these so they make sense
    # e.g. use some form of l1 regularization, add l2 reg, etc.
    # use more than 5 epochs (e.g. train until convergence)
    # there are other things to consider later, e.g. pytorch schedulers
    # (that change the learning rate over time)
    args.lr = 0.00005
    args.weight_decay = 0.0
    args.num_iter = 5 #num times through training data
    args.lam_l1=0.0
    args.stochastic = True

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args


def main():

    args = parse_config()
    # TODO (liang): add logging to this based on helpers/utils.py/log_this

    # TODO (max): make seeds work properly
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    
    Procgen_fn = lambda: ProcgenEnv(
        num_envs=1,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        rand_seed = seed
    )
    venv_fn = lambda: VecExtractDictObs(Procgen_fn(), "rgb")
    
    # here is where the T-REX procedure begins

    # collect a bunch of dems from trained models
    print('Generating demonstrations ...')
    #first we initialize the model of the correct shape using ppo.learn for 0 timesteps
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    policy_model = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0, seed = seed)
    dems = generate_procgen_dems(venv_fn, policy_model, args.models_dir, max_ep_len=512, num_dems=args.num_dems)

    print('Creating training data ...')
    num_snippets = args.num_snippets
    min_snippet_length = 10 #min length of trajectory for training comparison
    max_snippet_length = 100
    
    # TODO (anton): this process might be different depending on what the true reward model looks like
    # e.g. it's not very nice in coinrun
    training_data = create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length)


    # train a reward network using the dems collected earlier and save it
    print("Training reward model for", args.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = RewardTrainer(args, device)

    trainer.learn_reward(training_data)
    trainer.save_model()
    
    # print out predicted cumulative returns and actual returns

    with torch.no_grad():
        print('true     |predicted')
        for demo in sorted(dems, key = lambda x: x['return']):
            print(f"{demo['return']:9.2f}|{trainer.predict_traj_return(demo['observations']):9.2f}")


    print("accuracy", trainer.calc_accuracy(training_data))


if __name__=="__main__":
    main()
