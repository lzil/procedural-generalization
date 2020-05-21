import argparse
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from train_reward import RewardNet, get_file
import torch
import pandas as pd

from helpers.utils import get_id, filter_csv_pandas

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumspringgreen", "salmon"]) 
mpl.rcParams["font.family"] = "helvetica"

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='starpilot')
parser.add_argument('--mode', default='easy')
parser.add_argument('--sequential', type = int, default=0)

parser.add_argument('--demo_csv', default='trex/demos/demo_infos.csv')
parser.add_argument('--reward_csv', default='trex/reward_models/rm_infos.csv')
parser.add_argument('--reward_model', type = str)

args = parser.parse_args()

rm_id = get_id(args.reward_model)

# find the reward model and load it
path = glob.glob('./**/'+ rm_id + '.rm', recursive=True)[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = RewardNet().to(device)
net.load_state_dict(torch.load(path, map_location=torch.device(device)))
reward_function = lambda x: net.predict_batch_rewards(x)

# find the relevant demos and filter them
demo_infos = pd.read_csv(args.demo_csv)
demo_infos = demo_infos[demo_infos['env_name']==args.env_name]
demo_infos = demo_infos[demo_infos['mode']==args.mode]
demo_infos = demo_infos[demo_infos['sequential'] == args.sequential]
demo_infos = demo_infos[demo_infos['length'] > 100]
demo_infos = demo_infos[demo_infos['set_name'] == 'test']

# choose 12 of the demos at random to show
dems = []
for f_name in np.random.choice(demo_infos['demo_id'], 12):
    dems.append((f_name, get_file(f_name+'.demo')))


# get info about the reward model if csv is provided
rm_info = None
if args.reward_csv is not None:
    rm_infos = pd.read_csv(args.reward_csv)
    rm_infos = rm_infos[rm_infos['rm_id'] == rm_id]
    if rm_infos.shape[0] == 0:
        print(f'Reward model {rm_id} not found in given csv.')
    elif rm_infos.shape[0] == 1:
        rm_info = rm_infos.iloc[0]


fig, axs = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12, 7))

true_rews = dems[0][1]['rewards']
pred_rews = reward_function(dems[0][1]['observations'])
norm_const = np.sum(true_rews)/np.sum(pred_rews) 

for i, ax in enumerate(fig.axes):
    demo_id, demo = dems[i]
    true_rews = demo['rewards']
    pred_rews = reward_function(demo['observations'])

    # matplotlib formatting
    ax.set_title(demo_id)
    ax.axvline(x=0, color='dimgray', alpha = 1)
    ax.axhline(y=0, color='dimgray', alpha = 1)
    ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', color='white')

    # plot cumulative rewards for demonstration
    ax.plot(np.cumsum(true_rews), lw=2, label = 'true')
    ax.plot(np.cumsum(pred_rews) * norm_const, lw=2, label = 'predicted')

fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
fig.text(0.06, 0.5, 'cumulative reward', ha='center', va='center', rotation='vertical')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')
if rm_info is not None:
    fig.suptitle(f'reward model {rm_id}: {args.env_name}, {args.mode}, {args.sequential}; {rm_info.num_dems} dems', size='xx-large', weight='bold')
else:
    fig.suptitle(f'reward model {rm_id}: {args.env_name}, {args.mode}, {args.sequential}', size='xx-large', weight='bold')

plt.show()