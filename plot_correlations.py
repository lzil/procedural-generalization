import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import pdb
import pickle
import json

from reward_metric import get_corr_with_ground


reward_dir = 'trex/reward_models/reward_models_hard'
env_name = 'starpilot'


n_dems = [10, 15, 20, 30, 40]

def calc_correlations(n):

    dem_folder = env_name + str(n)

    with os.scandir(os.path.join(reward_dir, dem_folder)) as df:
        for entry in df:
            if entry.is_dir():
                models = os.listdir(entry)
                models.remove('reward_final.pth')
                models = [x.split('.pth')[0].split('_') for x in models]

                print('Organizing models')
                models = sorted(models, key=lambda x: (int(x[1]), int(x[2])))
                model_iterations = [int(x[1]) * 10000 + int(x[2]) for x in models]
                model_paths = ['_'.join(x) + '.pth' for x in models]

                pearsons = []
                spearmans = []
                for ix, path in enumerate(model_paths):
                    long_path = os.path.join(entry, path)
                    print('Processing ' + long_path)
                    pearson_r, spearman_r = get_corr_with_ground(long_path, env_name)

                    pearsons.append(pearson_r)
                    spearmans.append(spearman_r)

                with open(os.path.join(reward_dir, dem_folder, 'correlations.json')) as f:
                    json.dump({'model_its': model_iterations, 'model_paths': model_paths, 'pearsons': pearsons, 'spearmans': spearmans}, f)


def plot_correlations(n):
    dem_folder = env_name + str(n)
    with open(os.path.join(reward_dir, dem_folder, 'correlations.json')) as f:
        correlations = json.load(f)

    iterations = correlations['model_its']
    pearsons = correlations['pearsons']
    spearmans = correlations['spearmans']

    plt.plot(iterations, pearsons, '-', label='pearson')
    plt.plot(iterations, spearmans, '--', label='spearman')

    plt.title('correlation vs number of training steps')
    plt.xlabel('# steps')
    plt.ylabel('r')
    plt.legend()
    plt.show()




def main():
    calc_correlations(40)


if __name__ == '__main__':
    main()

                    

