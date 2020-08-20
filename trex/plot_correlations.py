import pandas as pd
import matplotlib.pyplot as plt
import argparse
from helpers.utils import filter_csv_pandas

from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='fruitbot')
parser.add_argument('--distribution_mode', default='easy')
parser.add_argument('--sequential', default='0', type=int)

parser.add_argument('--demo_csv', default='demos/demo_infos.csv')
parser.add_argument('--rm_csv_path', default='reward_models/rm_infos.csv')

# use length of demonstration as proxy for demonstration reward
parser.add_argument('--baseline_reward', action='store_true')

args = parser.parse_args()


# def get_baseline_corrs(demos):
#     pearson_r, pearson_p = pearsonr(demos['length'], demos['returns'])
#     spearman_r, spearman_p = spearmanr(demos['length'], demos['returns'])

#     return (pearson_r, spearman_r)


all_demos = pd.read_csv(args.demo_csv)
all_RMs = pd.read_csv(args.rm_csv_path)


reward_constraints = {
    'env_name': args.env_name,
    'mode': args.distribution_mode,
    'sequential': args.sequential
}

demo_constraints = {
    'set_name': 'test',
    'env_name': args.env_name,
    'mode': args.distribution_mode,
    'sequential': args.sequential
}

filtered_demos = filter_csv_pandas(all_demos, demo_constraints)
filtered_RMs = filter_csv_pandas(all_RMs, reward_constraints)

means = filtered_RMs.groupby('num_dems').mean()
stds = filtered_RMs.groupby('num_dems').std()

pearson_base, _ = pearsonr(filtered_demos['length'], filtered_demos['return'])
spearman_base, _ = spearmanr(filtered_demos['length'], filtered_demos['return'])
ax = means.plot(y=['spearman', 'pearson'], yerr=stds, kind='bar', capsize=4, figsize=(16, 9))
plt.axhline(y=pearson_base, label='pearson baseline', ls='--')
plt.axhline(y=spearman_base, color='salmon', label='spearman baseline', ls='--')
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=['pearson baseline', 'spearman baseline', 'spearman', 'pearson'])
plt.show()
