import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from train_reward import RewardNet, get_file
import torch
import pandas as pd



parser = argparse.ArgumentParser()

parser.add_argument('--env_name', default='starpilot')
parser.add_argument('--distribution_mode', default='easy')
parser.add_argument('--sequential', type = int, default=0)

parser.add_argument('--demo_csv_path', default='trex/demos/demo_infos.csv')
parser.add_argument('--reward_path', type = str)

args = parser.parse_args()

path = glob.glob('./**/'+args.reward_path +'.rm', recursive=True)[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = RewardNet().to(device)
net.load_state_dict(torch.load(path, map_location=torch.device(device)))
reward_function = lambda x: net.predict_batch_rewards(x)

demo_infos = pd.read_csv(args.demo_csv_path)
demo_infos = demo_infos[demo_infos['env_name']==args.env_name]
demo_infos = demo_infos[demo_infos['mode']==args.distribution_mode]
demo_infos = demo_infos[demo_infos['sequential'] == args.sequential]
demo_infos = demo_infos[demo_infos['length'] > 100]
# demo_infos = demo_infos[demo_infos['demo_id'] == '1_170_187_799']


dems = []
for f_name in np.random.choice(demo_infos['demo_id'],4):
    dems.append(get_file(f_name+'.demo'))


fig, axs = plt.subplots(2,2,sharex=True, sharey=True)

for i, ax in enumerate(fig.axes):
    demo = dems[i]
    true_rews = demo['rewards']
    pred_rews = reward_function(demo['observations'])

    ax.plot(np.cumsum(true_rews), label = 'true')
    ax.plot(np.cumsum(pred_rews), label = 'predicted')

    ax.legend()

plt.show()