import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import pdb
import pickle
import json
import csv

import argparse

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from helpers.utils import *
from reward_metric import get_corr_with_ground
from train_reward import get_demo


# takes in directory of reward models and demonstrations, and appropriate constraints,
# and calculates the correlations and saves them to a json file
def calc_correlations(reward_dir, demo_dir, r_constraints={}, d_constraints={}, max_set_size=200, save_path=None, verbose=False, baseline_reward=False):
    print('Calculating correlations of reward models.')

    # baseline reward is for using a proxy (currently demo length) as the reward, so don't need any in that case
    if baseline_reward:
        print('== Calculating baseline reward - completely skipping the use of reward models.')
    else:
        # figure out which reward models to use
        print(f'== r_constraints: {r_constraints}')
        rows = filter_csv(os.path.join(reward_dir, 'reward_model_infos.csv'), r_constraints)
        print(f'== Evaluating {len(rows)} reward models.')

    
    # figure out which demonstrations to use and load them
    print(f'== d_constraints: {d_constraints}')
    rows = filter_csv(os.path.join(demo_dir, 'demo_infos.csv'), d_constraints)
    demos = [get_demo(row['path']) for row in rows]
    print(f'== Using {len(demos)} demonstrations.')
    
    # do actual correlation calculations
    if baseline_reward:
        rs = []
        for dem in demos:
            # using length as the proxy
            r_prediction = len(dem['observations'])
            r_true = dem['return']

            rs.append((r_true, r_prediction))

        # calculate correlations and print them
        rs_by_var = list(zip(*rs))
        pearson_r, pearson_p = pearsonr(rs_by_var[0], rs_by_var[1])
        spearman_r, spearman_p = spearmanr(rs)

        ids = ['BASELINE_REWARD']
        pearsons = [pearson_r]
        spearmans = [spearman_r]

    else:
        pearsons = []
        spearmans = []
        ids = []
        for r in range(len(rows)):
            r_path = rows[r]['path']
            rm_id = get_id(r_path)
            print(f'{r+1}/{len(rows)}: {rm_id}, {r_path}')

            pearson_r, spearman_r = get_corr_with_ground(
                demos=demos,
                reward_path=r_path,
                verbose=verbose,
                baseline_reward=baseline_reward
            )

            ids.append(rm_id)
            pearsons.append(pearson_r)
            spearmans.append(spearman_r)

    infos = {}
    for idx, i in enumerate(ids):
        infos[i] = (pearsons[idx], spearmans[idx])

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(infos, f)
            print(f'== Saved correlations into {save_path}')

    return infos

# given correlations json (as produced in calc_correlations), plot it based on parameters
# used in training the reward model (hence the need for the reward csv)
def plot_correlations(infos, reward_dir, plot_type='num_dems', fig_path=None, show_fig=True):
    if plot_type == 'num_dems':
        # fix old infos. don't read too much into this block, it's kept for legacy reasons
        if 'ids' in infos:
            infos_ = {}
            for idx, i in enumerate(infos['ids']):
                infos_[i] = (infos['pearsons'][idx], infos['spearmans'][idx])
            infos = infos_

        print('Plotting correlations based on number of demonstrations.')
        ids = infos.keys()

        # filter reward models
        with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
            reader = csv.DictReader(master, delimiter=',')
            demo_bins = {}
            for row in reader:
                rm_id = get_id(row['path'])
                # reward model that we're not considering
                if rm_id not in ids:
                    continue
                # reward model trained with specific # demonstrations
                if row['num_dems'] not in demo_bins:
                    demo_bins[row['num_dems']] = [rm_id]
                else:
                    demo_bins[row['num_dems']].append(rm_id)

        print(f'Using {len(demo_bins)} demo bins.')

        # do some np magic to summarize data. i should probably learn pandas to do this
        demo_corrs_mean = []
        demo_corrs_all = []
        for k,v in demo_bins.items():
            pearson_r = []
            spearman_r = []
            for rm_id in v:
                pearson_r.append(infos[rm_id][0])
                spearman_r.append(infos[rm_id][1])
            p_k_avg = np.mean(pearson_r)
            p_k_std = np.std(pearson_r)
            s_k_avg = np.mean(spearman_r)
            s_k_std = np.std(spearman_r)
            demo_corrs_mean.append((int(k), p_k_avg, p_k_std, s_k_avg, s_k_std))
            demo_corrs_all.append((int(k), pearson_r, spearman_r))


        demo_corrs = sorted(demo_corrs_mean, key=lambda x: x[0])
        demo_corrs_T = list(zip(*demo_corrs))

        for d in demo_corrs_all:
            for j in range(len(d[1])):
                plt.plot(d[0], d[1][j], marker='o', ms=5, alpha=.3, color='skyblue')
                plt.plot(d[0], d[2][j], marker='o', ms=5, alpha=.3, color='salmon')

        plt.plot(demo_corrs_T[0], demo_corrs_T[1], marker='v', ms=8, ls='-', lw=3, color='skyblue', label='pearson')
        plt.plot(demo_corrs_T[0], demo_corrs_T[3], marker='^', ms=8, ls='--', lw=3, color='salmon', label='spearman')

        y_min = min(min(demo_corrs_T[1]), min(demo_corrs_T[3]))
        #plt.yticks(np.arange(y_min, 1.1, 0.1))
        plt.grid(which='both', axis='y')

        plt.title('reward model correlations', fontdict={'fontsize':15, 'fontweight':'bold'})
        plt.xlabel('# demonstrations', fontdict={'fontsize': 12})
        plt.ylabel('r', fontdict={'fontsize': 12})
        plt.ylim((-1, 1))
        plt.legend()

        if fig_path is not None:
            plt.savefig(fig_path)
        if show_fig:
            plt.gcf()
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=['calc', 'plot'], help='whether to calculate correlations, or load from saved file')

    parser.add_argument('--corr_path', default=None, help='path for the correlations file, only used in "plot" mode')

    parser.add_argument('--log_dir', default='trex/logs/corrs', help='directory to log the correlations')
    parser.add_argument('--fig_dir', default='trex/logs/corrs', help='directory for figures')

    parser.add_argument('--env_name', default='starpilot')
    parser.add_argument('--distribution_mode', default='easy')
    parser.add_argument('--sequential', default='0', type=str)

    parser.add_argument('--demo_dir', default='trex/demos')
    parser.add_argument('--reward_dir', default='trex/reward_models/')

    # use length of demonstration as proxy for demonstration reward
    parser.add_argument('--baseline_reward', action='store_true')

    args = parser.parse_args()

    return args

def main():
    # either calculate correlations from scratch, or just plot based on a saved correlations file

    args = parse_args()

    demo_constraints = {
        'set_name': 'test',
        'env_name': args.env_name,
        'mode': args.distribution_mode,
        'sequential': args.sequential
    }

    reward_constraints = {
        'env_name': args.env_name,
        'mode': args.distribution_mode,
        'sequential': args.sequential
    }

    args.d_constraints = demo_constraints
    args.r_constraints = reward_constraints

    # if not from file, then do the long correlations evaluation and save the results
    if args.mode == 'calc':
        run_dir, run_id = log_this(args, args.log_dir, log_name='', checkpoints=False)
        json_path = os.path.join(run_dir, f'correlations_{run_id}.json')
        infos = calc_correlations(
            reward_dir=args.reward_dir,
            demo_dir=args.demo_dir,
            r_constraints=reward_constraints,
            d_constraints=demo_constraints,
            save_path=json_path,
            baseline_reward=args.baseline_reward
        )
    # just pull the file
    elif args.mode == 'plot':
        with open(args.corr_path, 'r') as f:
            infos = json.load(f)
        json_path = args.corr_path

        fig_path = os.path.join(os.path.dirname(json_path), get_id(json_path) + '.png')
        plot_correlations(infos, args.reward_dir, fig_path=fig_path, show_fig=False)


if __name__ == '__main__':
    main()

                    

