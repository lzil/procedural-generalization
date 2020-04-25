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


n_dems = [10, 15, 20, 30, 40]

def calc_correlations(demos=None, tset='train', save_path=None):

    with open(os.path.join(reward_dir, 'reward_model_infos.csv')) as master:
        reader = csv.DictReader(master, delimiter=',')

        # filtering rows
        rows = []
        for row in reader:
            if demos is not None and demos != row['num_dems']:
                continue
            if tset is not None and tset != row['set']:
                continue
            rows.append(row)

    print(f'== Using {len(rows)} rows.')

    pearsons = []
    spearmans = []
    ids = []
    for r in range(len(rows)):
        path = rows[r]['path']
        rm_id = get_id(path)
        print(f'{r+1}/{len(rows)}: {rm_id}, {path}')

        pearson_r, spearman_r = get_corr_with_ground(path, env_name)

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
            reader = csv.reader(master, delimiter=',')

            # filtering rows
            demo_bins = {}
            for row in reader:
                rm_id = get_id(row['path'])
                if rm_id not in ids:
                    continue
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
    infos = calc_correlations(tset=None, save_path='correlations.json')

    plot_correlations(infos)


if __name__ == '__main__':
    main()

                    

