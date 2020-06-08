import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import scipy
import numpy as np
import pandas as pd

from helpers.affine import least_l2_affine


from train_reward import RewardNet, get_demo

class PotentialNet(nn.Module):
    """NN approximating a potential shaping funciton"""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16*16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x_d = x.permute(0,3,1,2).to(self.device)

        return self.model(x_d).flatten()

class EpicDataset(Dataset):
    """EPIC training dataset"""

    def __init__(self, dems, reward_function):

        """
        Args:
            dems : list of demonstrations 
            reward_function : learned proxy reward function
        """
    
        end_state = np.ones_like(dems[0]['observations'][0])
        self.s, self.a, self.s_1, self.Rs, self.Rt = [], [], [], [], []

        for demo in dems:
            self.s.extend(demo['observations'])
            self.a.extend(demo['actions'])
            demo['s_1'] = np.append(demo['observations'][1:], [end_state], axis = 0)
            self.s_1.extend(demo['s_1'])
            self.Rs.extend(reward_function(demo['observations']))            
            self.Rt.extend(demo['rewards'])

        assert len(self.s) == len(self.s_1) == len(self.Rs) == len(self.a) == len(self.Rt)

        self.s = np.array(self.s, dtype=np.float32)
        self.s_1 = np.array(self.s_1, dtype=np.float32)
        self.Rs = np.array(self.Rs)
        self.Rt = np.array(self.Rt)
        self.a = np.array(self.a)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'s': self.s[idx], 'a': self.a[idx], 's_1' : self.s_1[idx], 'Rs' : self.Rs[idx], 'Rt' :  self.Rt[idx]}

        return sample


def approximate_EPIC(reward_function, D):
    """Finds the approximate epic distance"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = EpicDataset(D, reward_function)
    dataloader = DataLoader(dataset, batch_size = 1024, shuffle=True, collate_fn = default_collate, pin_memory = True, drop_last = True)
    gamma = torch.tensor(0.99, dtype=torch.float32, device='cuda:0') 
    #initialize potential shaping funtion
    SH = PotentialNet().to(device)
    #find initaial nu and c that are close to optimal
    v, c  = least_l2_affine(dataset.Rs, dataset.Rt)
    v = torch.tensor(v, dtype=torch.float32, device=device, requires_grad=True) 
    c = torch.tensor(c, dtype=torch.float32, device=device, requires_grad=True) 
    
    l = nn.MSELoss(reduction = 'mean')
    optimizer = optim.Adam(list(SH.parameters())+[v, c], lr=0.002)
    print(f'total transitions = {len(dataset)}')

    print('Starting training')
    for epoch in range(200):
        epoch_losses = []
        for i, sampe_batch in enumerate(dataloader):
            optimizer.zero_grad()

            R_equiv =  torch.exp(v) * sampe_batch['Rs'].to(device) + c + gamma*SH(sampe_batch['s_1']) - SH(sampe_batch['s'])

            EPIC_loss = l(R_equiv, sampe_batch['Rt'].to(device))
            EPIC_loss.backward()

            optimizer.step()

            item_loss = EPIC_loss.item()
            epoch_losses.append(item_loss)

        print(f"epoch : {epoch}, EPIC distance : {np.mean(epoch_losses)}" )






def main():
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--env_name', default='starpilot')
    parser.add_argument('--distribution_mode', default='easy')
    parser.add_argument('--sequential', default=0)

    parser.add_argument('--demo_csv_path', default='trex/demos/demo_infos.csv')
    parser.add_argument('--reward_path', type = str)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RewardNet().to(device)
    net.load_state_dict(torch.load(args.reward_path, map_location=torch.device(device)))

    reward_function = lambda x: net.predict_batch_rewards(x)

    demo_infos = pd.read_csv(args.demo_csv_path)
    demo_infos = demo_infos[demo_infos['env_name']==args.env_name]
    demo_infos = demo_infos[demo_infos['mode']==args.distribution_mode]
    demo_infos = demo_infos[demo_infos['sequential'] == args.sequential]

    dems = []
    for f_name in demo_infos['path'][:200]:
        dems.append(get_demo(f_name))

    approximate_EPIC(reward_function, dems)



if __name__=="__main__":
    main()
