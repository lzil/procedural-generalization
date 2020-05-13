# adapted heavily from https://github.com/hiwonjoon/ICML2019-TREX/blob/master/atari/LearnAtariReward.py

import numpy as np
import pandas as pd
import csv
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import os, glob
import random
import sys
from shutil import copy2

import logging

import tensorflow as tf

# from baselines.ppo2 import ppo2
import helpers.baselines_ppo2 as ppo2 # use this for adjusted logging ability
from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

from helpers.trajectory_collection import ProcgenRunner, generate_procgen_dems
from helpers.utils import *

import argparse


log = None

from scipy.stats import pearsonr
from scipy.stats import spearmanr

def get_corr_with_ground(demos, net, verbose=False, baseline_reward=False):

    rs = []
    for dem in demos:
        if baseline_reward:
            r_prediction = len(dem['observations'])
        else:
            r_prediction = np.sum(net.predict_batch_rewards(dem['observations']))
        
        r_true = dem['return']

        rs.append((r_true, r_prediction))

    # calculate correlations and print them
    rs_by_var = list(zip(*rs))
    pearson_r, pearson_p = pearsonr(rs_by_var[0], rs_by_var[1])
    spearman_r, spearman_p = spearmanr(rs)

    if verbose:
        print(f'(pearson_r, spearman_r): {(pearson_r, spearman_r)}')

    return (pearson_r, spearman_r)


def create_training_data(dems, num_snippets, min_snippet_length, max_snippet_length, validation=True, verbose=True):
    """
    This function takes a set of demonstrations and produces 
    a training set consisting of pairs of clips with assigned preferences
    """

    #Print out some info
    if verbose:
        logging.info( f' {len(dems)} demonstrations provided')
        logging.info(f"demo lengths : {[d['length'] for d in dems]}")
        logging.info(f"demo returns : {[d['return'] for d in dems]}")
        demo_lens = [d['length'] for d in dems]
        logging.info(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')
        assert min_snippet_length < min(demo_lens), "One of the trajectories is too short"
    
    training_data = []
    validation_data = []
    # pick 2 of demos to be validation demos
    if validation:
        val_idx = np.random.choice(len(dems), int(len(dems)/6),  replace = False)

    while len(training_data) < num_snippets:

        #pick two random demos
        i1, i2 = np.random.choice(len(dems) ,2,  replace = False)

        if validation:
            is_validation = (i1 in val_idx) or (i2 in val_idx)
        else:
            is_validation = False
        # make d0['return'] <= d1['return']
        d0, d1 = sorted([dems[i1], dems[i2]], key = lambda x: x['return'])   
        if d0['return'] == d1['return']:
            continue
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


        ### This doesn't make any difference 

        # # randomize label so reward learning model won't learn heuristic
        # label = np.random.randint(2)
        # if label:
        #     training_data.append(([clip0, clip1], np.array([1])))
        # else:
        #     training_data.append(([clip1, clip0], np.array([0])))

    return np.array(training_data), np.array(validation_data)


# actual reward learning network
class RewardNet(nn.Module):
    def __init__(self, output_abs=False):
        super().__init__()
        self.output_abs = output_abs

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
        if self.output_abs:
            r = torch.abs(self.model(x))
        else:
            r = self.model(x)
        all_reward = torch.sum(r)
        all_reward_abs = torch.sum(torch.abs(r))
        return all_reward, all_reward_abs

    def predict_batch_rewards(self, batch_obs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            x = torch.tensor(batch_obs, dtype=torch.float32).permute(0,3,1,2).to(device) #get into NCHW format
            #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
            if self.output_abs:
                r = torch.abs(self.model(x))
            else:
                r = self.model(x) 
            return r.cpu().numpy().flatten()

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        all_r_i, abs_r_i = self.predict_returns(traj_i)
        all_r_j, abs_r_j = self.predict_returns(traj_j)
        return torch.stack((all_r_i, all_r_j)), abs_r_i + abs_r_j


# trainer wrapper in order to make training the reward model a neat process
class RewardTrainer:
    def __init__(self, args, device):
        self.device = device
        self.net = RewardNet(output_abs=args.output_abs).to(device)
        self.best_model = copy.deepcopy(self.net.state_dict())
        self.args = args

    # Train the network
    def learn_reward(self, train_set, val_set, test_set, test_dems):
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        max_val_acc = 0 
        eps_no_max = 0

        with open (self.args.debug_csv, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['n_train_samples','train_acc', 'val_acc','test_acc', 'pearson', 'spearman'])

            for epoch in range(self.args.max_num_epochs):
                epoch_loss = 0
                rewards = []
                avg_rewards = []
                np.random.shuffle(train_set)
                #each epoch consists of some updates - NOT passing through whole test set.
                for i, ([traj_i, traj_j], label) in enumerate(train_set[:self.args.epoch_size]):

                    ti = torch.from_numpy(traj_i).float().to(self.device)
                    tj = torch.from_numpy(traj_j).float().to(self.device)
                    lb = torch.from_numpy(label).to(self.device)

                    optimizer.zero_grad()

                    #forward + backward + optimize
                    outputs, abs_rewards = self.net.forward(ti, tj)

                    rewards.append(outputs[0].item())
                    rewards.append(outputs[1].item())
                    avg_rewards.append(abs_rewards.item())

                    outputs = outputs.unsqueeze(0)

                    # L1 regularization on the output
                    l1_reg = abs_rewards * self.args.lam_l1
                    
                    loss = loss_criterion(outputs, lb) + l1_reg
                    loss.backward()
                    optimizer.step()

                    item_loss = loss.item()
                    epoch_loss += item_loss
                
                train_acc = self.calc_accuracy(train_set[np.random.choice(len(train_set), size=100, replace=False)])
                val_acc = self.calc_accuracy(val_set[np.random.choice(len(val_set), size=100, replace=False)]) #keep validation set to 1000
                test_acc = self.calc_accuracy(test_set)
                pearson, spearman = get_corr_with_ground(test_dems, self.net)
                writer.writerow([epoch*self.args.epoch_size, train_acc, val_acc, test_acc, pearson, spearman])

                avg_reward = np.mean(np.array(rewards))
                avg_abs_reward = np.mean(np.array(abs_rewards))

                logging.info(f"n_samples: {epoch*self.args.epoch_size:6g} | loss: {epoch_loss:5.2f} | rewards: {avg_reward.item():5.2f}/{avg_abs_reward.item():.2f} | pc: {pearson:5.2f} | sc: {spearman:5.2f}")
                logging.info(f'   | train_acc: {train_acc:6.4f} | val_acc: {val_acc:6.4f} | test_acc : {test_acc:6.4f}')
                if val_acc > max_val_acc:
                    self.save_model()
                    max_val_acc = val_acc
                    eps_no_max = 0
                    accs = (train_acc, val_acc, test_acc, pearson, spearman)
                else:
                    eps_no_max += 1

                #Early stopping
                if eps_no_max >= self.args.patience:
                    logging.info(f'Early stopping after epoch {epoch}')
                    self.net.load_state_dict(self.best_model)  #loading the model with the best validation accuracy
                    break
                    

        logging.info("finished training")
        return os.path.join(self.args.checkpoint_dir, 'reward_best.pth'), accs

    # save the final learned model
    def save_model(self):
        torch.save(self.net.state_dict(), os.path.join(self.args.checkpoint_dir, 'reward_best.pth'))
        self.best_model = copy.deepcopy(self.net.state_dict())

    # calculate and return accuracy on entire training set
    def calc_accuracy(self, data):
        num_correct = 0.
        with torch.no_grad():
            for [traj_i, traj_j], label in data:
                ti = torch.from_numpy(traj_i).float().to(self.device)
                tj = torch.from_numpy(traj_j).float().to(self.device)
                lb = torch.from_numpy(label).to(self.device)

                #forward to get logits
                outputs, abs_return = self.net.forward(ti, tj)
                _, pred_label = torch.max(outputs,0)
                if pred_label.item() == lb:
                    num_correct += 1.
        return num_correct / len(data)


    # purpose of these two functions is to get predicted return (via reward net) from the trajectory given as input
    def predict_reward_sequence(self, traj):
        rewards_from_obs = []
        with torch.no_grad():
            for s in traj:
                r = self.net.predict_returns(torch.from_numpy(np.array([s])).float().to(self.device))[0].item()
                rewards_from_obs.append(r)
        return rewards_from_obs

    def predict_traj_return(self, traj):
        return sum(self.predict_reward_sequence(traj))

def parse_config():
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--seed', type = int, help="random seed for experiments")
    parser.add_argument('--sequential', type = int, default = 0, 
        help = '0 means not sequential, any other number creates sequential env with start_level = args.sequential')

    parser.add_argument('--num_dems',type=int, default = 12 , help = 'Number off demonstrations to train on')
    parser.add_argument('--max_return',type=float , default = 1.0, 
                        help = 'Maximum return of the provided demonstrations as a fraction of max available return')
    parser.add_argument('--num_snippets', default=50000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--min_snippet_length', default=20, type=int, help="Min length of tracjectory for training comparison")
    parser.add_argument('--max_snippet_length', default=100, type=int, help="Max length of tracjectory for training comparison")
    
    parser.add_argument('--epoch_size', default = 2000, type=int, help ='How often to measure validation accuracy')
    parser.add_argument('--max_num_epochs', type = int, default = 50, help = 'Number of epochs for reward learning')
    parser.add_argument('--patience', type = int, default = 6, help = 'early stopping patience')

    parser.add_argument('--lr', type=float, default=5e-5, help='reward model learning rate')
    parser.add_argument('--lam_l1', type=float, default=0, help='l1 penalization of abs value of output')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay of updates')
    parser.add_argument('--output_abs', action='store_true', help='absolute value the output of reward model')
    
    #trex/[folder to save to]/[optional: starting name of all saved models (otherwise just epoch and iteration)]
    parser.add_argument('--log_dir', default='trex/logs', help='general logs directory')
    # parser.add_argument('--log_name', default=None, help='specific name for this run')

    parser.add_argument('--log_to_file', action='store_true', help='print to a specific log file instead of console')

    parser.add_argument('--demo_folder', nargs='+', default=['trex/demos'], help='path to folders with demos')
    parser.add_argument('--demo_csv', nargs='+', default=['trex/demos/demo_infos.csv'], help='path to csv files with demo info')

    parser.add_argument('--save_dir', default='trex/reward_models', help='where the models and csv get stored')
    parser.add_argument('--save_name', default=None, help='suffix to the name of the csv/file folder for saving')
    
    args = parser.parse_args()

    args.lr = 0.00005
    args.weight_decay = 0.0
    
    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args

def store_model(state_dict_path, max_return, max_length, accs, args):

    csv_name = 'rm_infos.csv' if args.save_name is None else f'rm_infos_{args.save_name}.csv'
    info_path = os.path.join(args.save_dir, csv_name)

    if not os.path.exists(info_path):
        with open(info_path, 'w') as f: 
            rew_writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            rew_writer.writerow(['rm_id', 'method', 'env_name', 'mode',
                                 'num_dems', 'max_return', 'max_length',
                                'sequential', 'train_acc','val_acc',
                                'test_acc','pearson','spearman'])

    files_name = 'model_files' if args.save_name is None else f'model_files_{args.save_name}'
    model_dir = os.path.join(args.save_dir, files_name)
    os.makedirs(model_dir, exist_ok=True)

    rm_id = str(args.seed)[:3] + '_' + str(args.seed)[3:]
    save_path = os.path.join(model_dir, str(args.seed)[:3] + '_' + str(args.seed)[3:] + '.rm')
    copy2(state_dict_path, save_path)

    train_acc, val_acc, test_acc, pearson, spearman = accs
    with open(info_path, 'a') as f: 
        rew_writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rew_writer.writerow([rm_id, 'trex', args.env_name, args.distribution_mode,
                            args.num_dems, max_return, max_length, args.sequential,
                            train_acc, val_acc, test_acc, pearson, spearman])

def get_demo(file_name):
    #searches for the demo with the given name in all subfolders,
    #then loads it and returns 
    path = glob.glob('./**/'+file_name, recursive=True)[0]
    demo = pickle.load(open(path, 'rb'))

    return demo


def main():

    args = parse_config()

    # do seed creation before log creation
    print('Setting up logging and seed creation', flush=True)
    if args.seed:
        seed = args.seed 
    else:
        seed = random.randint(1e6,1e7-1)
        args.seed = seed
    rm_id = '_'.join([str(seed)[:3], str(seed)[3:]])

    log_path, checkpoint_dir, run_id = log_this(args, args.log_dir, 'rm-' + rm_id)
    args.run_id = run_id
    args.checkpoint_dir = checkpoint_dir

    args.debug_csv = os.path.join(args.log_dir, 'rm-' + rm_id, f'debug_rm_{run_id}.csv')
    args.log_path = log_path
    logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    logging.info(f'Reward model id: {rm_id}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    
    
    # here is where the T-REX procedure begins
    constraints = {
        'env_name': args.env_name,
        'mode': args.distribution_mode,
        'demo_min_len': args.min_snippet_length,
        'sequential': args.sequential
    }

    for path in args.demo_csv:
        train_rows = filter_csv_pandas(path, {**constraints, **{'set_name': 'train'}})
        test_rows = filter_csv_pandas(path, {**constraints, **{'set_name': 'test'}})

    logging.info(f'Filtered demos: {len(train_rows)} training demos available, {args.num_dems} requested')

    #acquiring test demos for correlations and test accuracy

    logging.info('Creating testing data ...')
    n_test_demos = 100
    demo_ids_n = np.random.choice(test_rows['demo_id'], n_test_demos)
    test_dems = []
    for dem in demo_ids_n:
        for folder in args.demo_folder:
            fpath = os.path.join(folder, dem + '.demo')
            if os.path.isfile(fpath):
                test_dems.append(get_demo(fpath))
                break

    test_set, _ = create_training_data(
        dems = test_dems,
        num_snippets = 1000,
        min_snippet_length = args.min_snippet_length,
        max_snippet_length = args.max_snippet_length,
        validation = False,
        verbose = False
    )

    logging.info('Creating training data ...')

    #implemening uniformish distribution of demo returns
    min_return = train_rows.min()['return']
    max_return = (train_rows.max()['return'] - min_return) * args.max_return + min_return

    rew_step  = (max_return - min_return)/ 4
    dems = []
    seeds = []
    while len(dems) < args.num_dems:

        high = min_return + rew_step 
        while (high <= max_return) and (len(dems) < args.num_dems):
            #crerate boundaries to pick the demos from, and filter demos accordingly
            low = high - rew_step
            filtered_dems = train_rows[(train_rows['return'] >= low) & (train_rows['return']<= high)]
            #make sure we have only unique demos
            new_seeds = filtered_dems[~filtered_dems['demo_id'].isin(seeds)]['demo_id']
            #choose random demo and append
            if len(new_seeds) > 0:
                chosen_seed = np.random.choice(new_seeds, 1).item()
                for folder in args.demo_folder:
                    fpath = os.path.join(folder, chosen_seed + '.demo')
                    if os.path.isfile(fpath):
                        seeds.append(chosen_seed)
                        dems.append(get_demo(fpath))
                        break
            high += rew_step
    
    max_demo_return = max([demo['return'] for demo in dems])
    max_demo_length = max([demo['length'] for demo in dems])

    training_data = create_training_data(
        dems = dems,
        num_snippets = args.num_snippets,
        min_snippet_length = args.min_snippet_length,
        max_snippet_length = args.max_snippet_length,
        validation = True,
        verbose = False
    )

    # train a reward network using the dems collected earlier and save it
    logging.info("Training reward model for %s ...", args.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = RewardTrainer(args, device)

    train_set, val_set = training_data
    state_dict_path, accs = trainer.learn_reward(train_set, val_set, test_set, test_dems)

    # print out predicted cumulative returns and actual returns

    with torch.no_grad():
        logging.info('true     |predicted')
        for demo in sorted(dems[:20], key = lambda x: x['return']):
            logging.info(f"{demo['return']:<9.2f}|{trainer.predict_traj_return(demo['observations']):>9.2f}")

    logging.info(f"Final train set accuracy {trainer.calc_accuracy(train_set[:5000])}")

    store_model(state_dict_path, max_demo_return, max_demo_length, accs, args)


if __name__=="__main__":
    main()
