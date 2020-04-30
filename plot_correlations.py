import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import pdb
import pickle
import json
import csv

from reward_metric import get_corr_with_ground


reward_dir = 'trex/reward_models/'
env_name = 'starpilot'

demos_folder = 'trex/demos'


n_dems = [10, 15, 20, 30, 40]

def calc_correlations(r_constraints={}, save_path=None, verbose=True):

    with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
        reader = csv.DictReader(master, delimiter=',')

        # filtering rows
        rows = []
        for row in reader:
            skip = False
            for k,v in r_constraints.items():
                if row[k] != v:
                    skip = True
            if skip:
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

        d_constraints = {
            'set_name': 'TEST',
            'env_name': 'starpilot'
        }

        pearson_r, spearman_r = get_corr_with_ground(
            demos_folder=demos_folder,
            reward_path=r_path,
            constraints=d_constraints,
            verbose=False
        )

        ids.append(rm_id)
        pearsons.append(pearson_r)
        spearmans.append(spearman_r)

    infos = {
        'ids': ids,
        'pearsons': pearsons,
        'spearmans': spearmans
    }

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(infos, f)
            print(f'== Saved correlations into {save_path}')

    return (ids, pearsons, spearmans)


def get_id(path):
    # extract id from the path. a bit hacky but should get the job done
    rm_id = '.'.join(path.split('/')[-1].split('.')[:-1])
    return rm_id


def plot_correlations(infos, plot_type='num_dems'):
    if plot_type == 'num_dems':
        ids, pearsons, spearmans = infos

        with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
            reader = csv.DictReader(master, delimiter=',')

            # filtering rows
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

        demo_corrs = []
        for k,v in demo_bins.items():
            pearson_k = list(filter(lambda x: x in v, pearsons))
            spearman_k = list(filter(lambda x: x in v, spearmans))
            p_k_avg = np.mean(pearson_k)
            s_k_avg = np.mean(spearman_k)
            demo_corrs.append((int(k), p_k_avg, s_k_avg))

        demo_corrs = sorted(demo_corrs, key=lambda x: x[0])
        demo_corrs_T = list(zip(*demo_corrs))


        plt.plot(demo_corrs_T[0], demo_corrs_T[1], '-', label='pearson')
        plt.plot(demo_corrs_T[0], demo_corrs_T[2], '--', label='spearman')

        plt.title('correlation vs number of training steps')
        plt.xlabel('# steps')
        plt.ylabel('r')
        plt.legend()
        plt.show()



def main():
    reward_constraints = {
        'env_name': 'starpilot',
        'mode': 'easy',
        'sequential': '0.0'
    }
    infos = calc_correlations(r_constraints=reward_constraints, save_path='correlations.json')

    #plot_correlations(infos)


if __name__ == '__main__':
    main()

                    

