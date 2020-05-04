import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import pdb
import pickle
import json
import csv

from reward_metric import get_corr_with_ground, retain_row

def calc_correlations(reward_dir, demo_dir, r_constraints={}, d_constraints={}, save_path=None, verbose=True):
    print('Calculating correlations of reward models.')
    print(f'== Constraints: {r_constraints}')

    with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
        reader = csv.DictReader(master, delimiter=',')

        # filtering rows
        rows = []
        for row in reader:
            if not retain_row(row, r_constraints):
                continue
            rows.append(row)

    print(f'== Using {len(rows)} rows.')
    
    pearsons = []
    spearmans = []
    ids = []
    for r in range(len(rows)):
        r_path = rows[r]['path']
        rm_id = get_id(r_path)
        print(f'{r+1}/{len(rows)}: {rm_id}, {r_path}')

        pearson_r, spearman_r = get_corr_with_ground(
            demos_folder=demo_dir,
            reward_path=r_path,
            constraints=d_constraints,
            verbose=False
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


def get_id(path):
    # extract id from the path. a bit hacky but should get the job done
    rm_id = '.'.join(path.split('/')[-1].split('.')[:-1])
    return rm_id


def plot_correlations(infos, reward_dir, r_constraints, plot_type='num_dems'):
    if plot_type == 'num_dems':
        # fix old infos
        if 'ids' in infos:
            infos_ = {}
            for idx, i in enumerate(infos['ids']):
                infos_[i] = (infos['pearsons'][idx], infos['spearmans'][idx])
            infos = infos_

        print(f'== Using correlations: {infos}')
        ids = infos.keys()

        with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
            reader = csv.DictReader(master, delimiter=',')

            # filtering rows by demonstration count
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

        # plt.errorbar(demo_corrs_T[0], demo_corrs_T[1], yerr=demo_corrs_T[2], elinewidth=3, capsize=5, marker='v', ms=10, ls='-', lw=3, color='skyblue', label='pearson')
        # plt.errorbar(demo_corrs_T[0], demo_corrs_T[3], yerr=demo_corrs_T[4], elinewidth=2, capsize=3, marker='^', ms=10, ls='--', lw=3, color='salmon', label='spearman')

        #fig = plt.figure()

        for d in demo_corrs_all:
            for j in range(len(d[1])):
                plt.plot(d[0], d[1][j], marker='o', ms=5, alpha=.3, color='skyblue')
                plt.plot(d[0], d[2][j], marker='o', ms=5, alpha=.3, color='salmon')

        plt.plot(demo_corrs_T[0], demo_corrs_T[1], marker='v', ms=8, ls='-', lw=3, color='skyblue', label='pearson')
        plt.plot(demo_corrs_T[0], demo_corrs_T[3], marker='^', ms=8, ls='--', lw=3, color='salmon', label='spearman')

        plt.yticks(np.arange(-1.1, 1.1, 0.1))
        plt.grid(which='both', axis='y')

        plt.title(f'reward model correlations \n {r_constraints["env_name"]} sequential:{r_constraints["sequential"]}', fontdict={'fontsize':15, 'fontweight':'bold'})
        plt.xlabel('# demonstrations', fontdict={'fontsize': 12})
        plt.ylabel('r', fontdict={'fontsize': 12})
        plt.ylim((-1, 1))
        plt.legend()
        plt.savefig('figures/corrs_tmp.png')
        plt.gcf()
        plt.show()



def main():
    # either calculate correlations from scratch, or just plot based on a saved correlations file

    plot_mode = 'corrs'
    #plot_mode = 'plot'

    env_name = 'fruitbot'
    mode = 'easy'
    sequential = '0'


    reward_dir = 'trex/reward_models/'
    demo_dir = 'trex/fruit_dems'

    demo_constraints = {
        'set_name': 'TEST',
        'env_name': env_name,
        'mode': mode,
        'sequential': sequential
    }

    reward_constraints = {
        'env_name': env_name,
        'mode': mode,
        'sequential': sequential
    }

    # set up path of correlations json
    correlations_name = 'correlations_3.json'
    corrs_path = os.path.join('trex', 'logs', 'corrs')
    os.makedirs(corrs_path, exist_ok=True)
    correlations_path = os.path.join(corrs_path, correlations_name)

    if plot_mode == 'corrs':
        infos = calc_correlations(
            reward_dir=reward_dir,
            demo_dir=demo_dir,
            r_constraints=reward_constraints,
            d_constraints=demo_constraints,
            save_path=correlations_path
        )
    # don't bother plotting, just pull the file
    elif plot_mode == 'plot':
        with open(correlations_path, 'r') as f:
            infos = json.load(f)

    plot_correlations(infos, reward_dir, reward_constraints)


if __name__ == '__main__':
    main()

                    

