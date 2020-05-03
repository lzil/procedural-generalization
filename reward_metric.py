import numpy as np
import torch
import pandas as pd 
import os
import csv
import pickle
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from procgen import ProcgenEnv
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

import pdb
import argparse

from train_reward import RewardNet, generate_procgen_dems
from helpers.utils import add_yaml_args
from helpers.ProxyRewardWrapper import ProxyRewardWrapper
import helpers.baselines_ppo2 as ppo2



def parse_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--reward_path', type=str, default='trex/reward_models/starpilot10/checkpoints_788428/reward_final.pth')
    parser.add_argument('--env_name', type=str, default='starpilot')
    parser.add_argument('--distribution_mode', type=str, default='easy',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_dems', default=20, type=int, help="number of trajectories to use")
    # todo: save a bunch of test trajectories somewhere so these don't need to be generated from scratch every time

    parser.add_argument('--models_dir', default='trex/chaser_model_dir')
    # parser.add_argument('--traj_len')

    args = parser.parse_args()

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args

def main():

    args = parse_config()

    print(f'Evaluating reward model at: {args.reward_path} on {args.num_dems} trajectories.')

    
    #read the demo infos, see first 5 entries
    demo_infos = pd.read_csv('trex/demos/demo_infos.csv')
    demo_infos = demo_infos[demo_infos['set_name']=='TEST']
    demo_infos = demo_infos[demo_infos['env_name']==args.env_name]
    demo_infos = demo_infos[demo_infos['mode']==args.distribution_mode]
    # print(demo_infos.head())

    #unpickle just the entries where return is more then 10
    #append them to the dems list (100 dems)
    dems = []
    for path in demo_infos[demo_infos['return'] > 20]['path'][:500]:
        dems.append(pickle.load(open(path, "rb")))
    print(len(dems))
        
    # load learned reward model
    net = RewardNet()
    net.load_state_dict(torch.load(args.reward_path))

    rs = []
    for dem in dems:
        r_prediction = np.sum(net.predict_batch_rewards(dem['observations']))
        r_true = dem['return']

        rs.append((r_true, r_prediction))

    # calculate correlations and print them
    rs_by_var = list(zip(*rs))
    pearson_r, pearson_p = pearsonr(rs_by_var[0], rs_by_var[1])
    spearman_r, spearman_p = spearmanr(rs)

    print(f'Pearson r: {pearson_r}; p-val: {pearson_p}')
    print(f'Spearman r: {spearman_r}; p-val: {spearman_p}')


def get_corr_with_ground(demos_folder, reward_path, max_set_size=200, constraints={}, verbose=True):
    # setting up constraints on what demos we want
    if verbose:
        print(f'Using constraints: {constraints}')

    dems = []

    with open(os.path.join(demos_folder, 'demo_infos.csv')) as master:
        reader = csv.DictReader(master, delimiter=',')
        for row in reader:

            # making sure constraints are satisfied
            if not retain_row(row, constraints):
                continue

            dems.append(pickle.load(open(row['path'], "rb")))
            # limit the total number of demonstrations we compute correlation on
            if len(dems) >= max_set_size:
                break

        if verbose:
            print(f'Loaded {len(dems)} demonstrations.')
            
    # load learned reward model
    net = RewardNet()
    torch.load(reward_path, map_location=torch.device('cpu'))
    net.load_state_dict(torch.load(reward_path, map_location=torch.device('cpu')))
    rs = []
    for dem in dems:
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


def retain_row(row, constraints):
    for k,v in constraints.items():
        # respect maximum return constraints
        if 'demo_max_return' in constraints:
            if float(row['return']) > float(constraints['demo_max_return']):
                return False
        if 'rm_max_return' in constraints:
            if float(row['max_return']) > float(constraints['rm_max_return']):
                return False

        # all other constraints
        if row[k] != v:
            return False
    return True


if __name__ == '__main__':
    main()