# adapted heavily from https://github.com/hiwonjoon/ICML2019-TREX/blob/master/atari/LearnAtariReward.py

import numpy as np
import pandas as pd
import csv
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import os
import random
import sys

import tensorflow as tf

# from baselines.ppo2 import ppo2
import helpers.baselines_ppo2 as ppo2 # use this for adjusted logging ability
from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

from helpers.trajectory_collection import ProcgenRunner, generate_procgen_dems
from helpers.utils import *



import argparse

#traj_j in learn_reward and calc_accuracy into a function


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
    validation_data = []
    # pick 2 of demos to be validation demos
    val_idx = np.random.choice(len(dems), 2 , replace = False)
    for n in range(num_snippets):

        #pick two random demos
        i1, i2 = np.random.choice(len(dems) ,2, replace = False) 
        is_validation  = (i1 in val_idx) or (i2 in val_idx)   
        # d1['return'] <= d2['return']
        d0, d1 = sorted([dems[i1], dems[i2]], key = lambda x: x['return'])
        
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

        if is_validation:
            validation_data.append(([clip0, clip1], np.array([1])))
        else:
            training_data.append(([clip0, clip1], np.array([1])))

        # # randomize label so reward learning model won't learn heuristic
        # label = np.random.randint(2)
        # if label:
        #     training_data.append(([clip0, clip1], np.array([1])))
        # else:
        #     training_data.append(([clip1, clip0], np.array([0])))
    print(len(training_data), len(validation_data))
    return np.array(training_data), np.array(validation_data)


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
        
        training_set, validation_set = training_data

        max_val_acc = 0 
        eps_no_max = 0
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            np.random.shuffle(training_set)
            #each epoch consists of 5000 updates - NOT passing through whole test set.
            for i, ([traj_i, traj_j], label) in enumerate(training_set[:5000]):

                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)
                label = torch.from_numpy(label).to(self.device)

                optimizer.zero_grad()

                #forward + backward + optimize
                outputs, abs_rewards = self.net.forward(traj_i, traj_j)

                outputs = outputs.unsqueeze(0)

                # TODO: consider l2 regularization?
                #included with the optimizer weight_decay value
                #https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam 


                l1_reg = torch.tensor(0., requires_grad=True, device = self.device)
                for name, param in self.net.named_parameters():
                    if 'weight' in name:
                        l1_reg = l1_reg + torch.norm(param, 1)
                #print('l1 reg 1', l1_reg)
                #print('lam', self.args.lam_l1)
                l1_reg = abs_rewards * self.args.lam_l1

                #print('loss crit', loss_criterion(outputs, labels))
                #print('l1 reg', l1_reg)
                
                loss = loss_criterion(outputs, label) + l1_reg
                loss.backward()
                optimizer.step()

                item_loss = loss.item()
                epoch_loss += item_loss
                
            val_acc = self.calc_accuracy(validation_set[:1000]) #keep validation set under 1000 samples
            print(f"epoch : {epoch},  loss : {epoch_loss:6.2f}, val accuracy : {val_acc:6.4f}, abs_rewards : {abs_rewards.item():5.2f}")

            if val_acc > max_val_acc:
                self.save_model()
                max_val_acc = val_acc
                eps_no_max = 0
            else:
                eps_no_max += 1

            #Early stopping
            if eps_no_max >= self.args.patience:
                break
                print(f'Early stopping after epoch {epoch}')
            
        print("finished training")

    # save the final learned model
    def save_model(self):
        torch.save(self.net.state_dict(), os.path.join(self.args.checkpoint_dir, 'reward_final.pth'))

    # calculate and return accuracy on entire training set
    def calc_accuracy(self, training_data):
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
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

    parser.add_argument('--env_name', type=str, default='starpilot')
    parser.add_argument('--distribution_mode', type=str, default='hard',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_snippets', default=1000, type=int, help="number of short subtrajectories to sample")
    #trex/[folder to save to]/[optional: starting name of all saved models (otherwise just epoch and iteration)]
    parser.add_argument('--log_dir', default='trex/reward_models', help='general logs directory')
    parser.add_argument('--log_name', default='', help='specific name for this run')
    parser.add_argument('--demo_infos', default = 'trex/demos/starpilot_easy_demo_infos.csv',
     help="path to csv with trajectory infos")
    parser.add_argument('--num_dems',type=int, default = 6 , help = 'Number of demonstrations to train on')
    parser.add_argument('--max_dem_len',type=int, default = 3000 , help = 'Maximimum exprert demonstration length')
   
    parser.add_argument('--num_epochs', type = int, default = 20, help = 'Number of epochs for reward learning')
    args = parser.parse_args()

    args.lr = 0.00005
    args.weight_decay = 0.0
    args.lam_l1=0
    args.patience = 2
    args.stochastic = True

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args

def store_model(model, args):
    model_path = os.path.join('trex/reward_models', str(time.time()) + '.rm')

    with open(model_path, 'w') as f:
        
        fnames = ['first_name', 'last_name']
        writer = csv.DictWriter(f, fieldnames=fnames)    

        writer.writeheader()
        writer.writerow({'first_name' : 'John', 'last_name': 'Smith'})
        writer.writerow({'first_name' : 'Robert', 'last_name': 'Brown'})
        writer.writerow({'first_name' : 'Julia', 'last_name': 'Griffin'})

def main():

    args = parse_config()
    run_dir, checkpoint_dir, run_id = log_this(args, args.log_dir, args.log_name)
    args.checkpoint_dir = checkpoint_dir

    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
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


    demo_infos = pd.read_csv(args.demo_infos, index_col=0)


    #unpickle just the entries where return is more then 10
    #append them to the dems list (100 dems)
    dems = []
    for path in np.random.choice(demo_infos['path'], args.num_dems):
        dems.append(pickle.load(open(path, "rb")))
    
    print('Creating training data ...')
    num_snippets = args.num_snippets
    min_snippet_length = 20 #min length of tracjectory for training comparison
    max_snippet_length = 100
    
    # TODO (anton): this process might be different depending on what the true reward model looks like
    # e.g. it's not very nice in coinrun
    training_data= create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length)


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
            print(f"{demo['return']:<9.2f}|{trainer.predict_traj_return(demo['observations']):>9.2f}")


    print("Final train set accuracy", trainer.calc_accuracy(training_data[0]))


if __name__=="__main__":
    main()
