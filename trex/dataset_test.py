import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt

from train_reward import create_dataset, get_file

path = 'trex/demos/demo_infos_fruitbot.csv'
all_rows = pd.read_csv(path)

# rows = all_rows.sort_values('length', ascending=False)

# rows = rows.head(n=10)

# dems = []
# for dem in rows['demo_id']:
#     dems.append(get_file(dem + '.demo'))

# for dem in dems:
#     plt.imshow(dem['observations'][-2])
#     plt.show()

num_dems = 500

train_rows = all_rows[all_rows['set_name'] == 'train']

min_return = train_rows.min()['return']
max_return = (train_rows.max()['return'] - min_return) * 1 + min_return

rew_step = (max_return - min_return)/ 4
dems = []
seeds = []
while len(dems) < num_dems:

    high = min_return + rew_step 
    while (high <= max_return) and (len(dems) < num_dems):
        #crerate boundaries to pick the demos from, and filter demos accordingly
        low = high - rew_step
        filtered_dems = train_rows[(train_rows['return'] >= low) & (train_rows['return'] <= high)]
        #make sure we have only unique demos
        new_dems = filtered_dems[~filtered_dems['demo_id'].isin(seeds)]
        new_seeds = filtered_dems[~filtered_dems['demo_id'].isin(seeds)]['demo_id']
        #choose random demo and append
        if len(new_seeds) > 0:
            chosen_seed = np.random.choice(new_seeds, 1).item()
            print(new_dems[new_dems['demo_id'] == chosen_seed]['return'])

            dems.append(chosen_seed)
            seeds.append(chosen_seed)
        high += rew_step

pdb.set_trace()