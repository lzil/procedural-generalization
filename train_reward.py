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


def generate_procgen_dems(env_fn, model, model_dir, eps_per_model):
    dems = []
    # load all the models from a particular directory
    # to get episodes of varying reward, and add them to dems
    for model_file in os.scandir(model_dir):
        model.load(model_file)
        # TODO (anton): go through procgenrunner and make it simpler and more interpretable? get rid of useless parts
        collector = ProcgenRunner(env_fn, model, 512)
        dems.extend(collector.collect_episodes(eps_per_model))

    return dems


# TODO (max): use a DataLoader for this entire process. a lot neater
def create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length):
    # collect training data
    # dems should be sorted by increasing returns

    #print out some info
    print(len(dems), ' demonstrations provided')
    print("demo lengths :", [d['length'] for d in dems])
    print('demo returns :', [d['return'] for d in dems])
    demo_lens = [d['length'] for d in dems]
    print(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')

    # TODO (anton): create a validation set as well. use a train dataloader and a separate test dataloader
    training_obs = []
    training_labels = []
    
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
            training_obs.append((clip0, clip1))
        else:
            training_obs.append((clip1, clip0))
        training_labels.append(label)

    return training_obs, training_labels


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
    def learn_reward(self, training_inputs, training_outputs):
        loss_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        cum_loss = 0.0
        training_data = list(zip(training_inputs, training_outputs))
        for epoch in range(self.args.num_iter):
            # TODO (max): train until convergence
            # TODO (max): use a DataLoader here
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):
                traj_i, traj_j = training_obs[i]
                labels = np.array([training_labels[i]])
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)
                labels = torch.from_numpy(labels).to(self.device)

                optimizer.zero_grad()

                #forward + backward + optimize
                outputs, abs_rewards = self.net.forward(traj_i, traj_j)
                outputs = outputs.unsqueeze(0)
                # TODO: confirm the dimensionality here is correct. not totally sure
                # TODO: consider l2 regularization?
                loss = loss_criterion(outputs, labels) + self.args.lam_l1 * abs_rewards
                loss.backward()
                optimizer.step()

                item_loss = loss.item()
                cum_loss += item_loss
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
    def calc_accuracy(self, training_inputs, training_outputs):
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            # TODO: use a DataLoader
            for i in range(len(training_inputs)):
                label = training_outputs[i]
                traj_i, traj_j = training_inputs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)

                #forward to get logits
                outputs, abs_return = self.net.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs,0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_inputs)


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
    parser.add_argument('--num_envs', type=int, default=2, help="number of demos per model")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_snippets', default=6000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--models_dir', default = "trex/chaser_model_dir", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--reward_model_path', default='trex/reward_model_chaser', help="name and location for learned model params, e.g. ./learned_models/breakout.params")

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

    
    Procgen_fn = lambda: ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    venv_fn = lambda: VecExtractDictObs(Procgen_fn(), "rgb")
    
    # here is where the T-REX procedure begins

    # collect a bunch of dems from trained models
    print('Generating demonstrations ...')
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    policy_model = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0)
    dems = generate_procgen_dems(venv_fn, policy_model, args.models_dir, eps_per_model = args.num_envs)
    # TODO: why is args.num_envs used as a placeholder for eps_per_model?

    print('Creating training data ...')
    num_snippets = args.num_snippets
    min_snippet_length = 10 #min length of trajectory for training comparison
    max_snippet_length = 100
    
    # TODO (anton): this process might be different depending on what the true reward model looks like
    # e.g. it's not very nice in coinrun
    training_obs, training_labels = create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))

   
    # train a reward network using the dems collected earlier and save it
    print("Training reward model for", args.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = RewardTrainer(args, device)

    trainer.learn_reward(training_obs, training_labels)
    trainer.save_model()
    
    # print out predicted cumulative returns and actual returns

    with torch.no_grad():
        pred_returns = [trainer.predict_traj_return(traj) for traj in dems]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", trainer.calc_accuracy(training_obs, training_labels))


if __name__=="__main__":
    main()
