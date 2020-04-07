# adapted heavily from https://github.com/hiwonjoon/ICML2019-TREX/blob/master/atari/LearnAtariReward.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
import os

import tensorflow as tf

from baselines.ppo2 import ppo2
from procgen import ProcgenEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env import VecExtractDictObs
from baselines.common.models import build_impala_cnn

from helpers.trajectory_collection import ProcgenRunner
from helpers.utils import *

import argparse

#traj_j in learn_reward and calc_accuracy into a function


def generate_procgen_demonstrations(env_fn, model, model_dir, eps_per_model):
    eps = []
    # load all the models from a particular directory
    # to get episodes of varying reward, and add them to eps
    for model_file in os.scandir(model_dir):
        model.load(model_file)
        # TODO (anton): go through procgenrunner and make it simpler and more interpretable? get rid of useless parts
        collector = ProcgenRunner(env_fn, model, 512)
        eps.extend(collector.collect_episodes(eps_per_model))

    demonstrations = [e['observations'] for e in eps]
    learning_returns = [e['return'] for e in eps]
    learning_rewards = [e['rewards'] for e in eps]

    return demonstrations, learning_returns, learning_rewards


# TODO (max): use a DataLoader for this entire process. a lot neater
def create_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length):
    # collect training data
    # demonstrations should be sorted by increasing returns

    # TODO (anton): create a validation set as well. use a train dataloader and a separate test dataloader
    max_traj_len = 0
    training_obs = []
    training_labels = []

    n_demos = len(demonstrations)
    demo_lens = [len(t) for t in demonstrations]
    print(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns, ti < tj
        ti, tj = np.sort(np.random.choice(n_demos, 2, replace = False))
        
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        cur_min_len = min(len(demonstrations[ti]), len(demonstrations[tj]))
        cur_max_snippet_len = min(cur_min_len, max_snippet_length)
        cur_len = np.random.randint(min_snippet_length, cur_max_snippet_len)

        #pick tj snippet to be later than ti
        ti_start = np.random.randint(cur_min_len - cur_len + 1)
        tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - cur_len + 1)

        traj_i = demonstrations[ti][ti_start:ti_start+cur_len]
        traj_j = demonstrations[tj][tj_start:tj_start+cur_len]

        # update global maximum trajectory length
        max_traj_len = max(max_traj_len, len(traj_i))

        # randomize label so reward learning model won't learn heuristic
        label = np.random.randint(2)
        if label:
            training_obs.append((traj_i, traj_j))
        else:
            training_obs.append((traj_j, traj_i))
        training_labels.append(label)

    print(f"max traj length: {max_traj_len}")
    return training_obs, training_labels


# actual reward learning network
class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()

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

    def predict_returns(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        r = self.model(x)
        all_reward = torch.sum(r)
        all_reward_abs = torch.sum(torch.abs(r))
        return all_reward, all_reward_abs

    def predict_batch_rewards(self, batch_obs):
        with torch.no_grad():
            x = torch.tensor(batch_obs, dtype=torch.float32).permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
            r = self.model(x)
            return r.numpy().flatten()

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        all_r_i, abs_r_i = self.predict_returns(traj_i)
        all_r_j, abs_r_j = self.predict_returns(traj_j)
        return torch.stack((all_r_i, all_r_j)), abs_r_i + abs_r_j


# trainer wrapper in order to make training the reward model a neat process
class RewardTrainer:
    def __init__(self, args, device):
        self.device = device
        self.net = RewardNet().to(device)

        self.args = args

    # Train the network
    def learn_reward(self, training_inputs, training_outputs):
        loss_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        cum_loss = 0.0
        epoch_loss = 0.0
        training_data = list(zip(training_inputs, training_outputs))
        for epoch in range(self.args.num_iter):
            # TODO (max): train until convergence
            # TODO (max): use a DataLoader here
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):
                traj_i, traj_j = training_obs[i]
                labels = np.array([training_labels[i]])
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)
                labels = torch.from_numpy(labels).to(self.device)

                optimizer.zero_grad()

                #forward + backward + optimize
                outputs, abs_rewards = self.net.forward(traj_i, traj_j)
                #print('outputs', outputs)
                #print('abs_rewards', abs_rewards)
                outputs = outputs.unsqueeze(0)
                #print('outputs unsqueezed', outputs)
                #print('labels', labels)
                # TODO: confirm the dimensionality here is correct. not totally sure
                #looks good

                # TODO: consider l2 regularization?
                #included with the optimizer weight_decay value
                #https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam 
                
                #l1_reg = self.args.lam_l1 * abs_rewards
                #implemented l1 reg using weights, above is alt way?

                l1_reg = torch.tensor(0., requires_grad=True)
                for name, param in self.net.named_parameters():
                    if 'weight' in name:
                        l1_reg = l1_reg + torch.norm(param, 1)
                #print('l1 reg 1', l1_reg)
                #print('lam', self.args.lam_l1)
                l1_reg = l1_reg * self.args.lam_l1

                #print('loss crit', loss_criterion(outputs, labels))
                #print('l1 reg', l1_reg)
                
                loss = loss_criterion(outputs, labels) + l1_reg
                loss.backward()
                optimizer.step()

                item_loss = loss.item()
                epoch_loss += item_loss
                #print(item_loss)
                #print(cum_loss)
                if i % 1000 == 999:
                    cum_loss += epoch_loss
                    print("epoch {}, step {}: loss {}".format(epoch, i, cum_loss))
                    print(f'absolute rewards = {abs_rewards.item()}')
                    # TODO: give this a different name for each log so it doesn't keep overwriting                  
                    torch.save(self.net.state_dict(), self.args.reward_model_path + '_' + str(epoch) + '_' + str(i))
                    if (1 - (cum_loss-epoch_loss)/cum_loss) < self.args.converg: #convergence
                        break
                    epoch_loss = 0.0

            # TODO (max): might want to calculate absolute accuracy every epoch or so

        print("finished training")

    # save the final learned model
    def save_model(self):
        torch.save(self.net.state_dict(), self.args.reward_model_path)

    # calculate and return accuracy on entire training set
    def calc_accuracy(self, training_inputs, training_outputs):
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            # TODO: use a DataLoader
            for i in range(len(training_inputs)):
                label = training_outputs[i]
                traj_i, traj_j = training_inputs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(self.device)
                traj_j = torch.from_numpy(traj_j).float().to(self.device)

                #forward to get logits
                outputs, abs_return = self.net.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs,0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_inputs)


    # purpose of these two functions is to get predicted return (via reward net) from the trajectory given as input
    def predict_reward_sequence(self, traj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rewards_from_obs = []
        with torch.no_grad():
            for s in traj:
                r = self.net.predict_returns(torch.from_numpy(np.array([s])).float().to(device))[0].item()
                rewards_from_obs.append(r)
        return rewards_from_obs

    def predict_traj_return(self, traj):
        return sum(self.predict_reward_sequence(traj))

def parse_config():
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='chaser')
    parser.add_argument('--distribution_mode', type=str, default='hard',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_envs', type=int, default=2, help="number of demos per model")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_snippets', default=6000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--models_dir', default = "trex/chaser_model_dir", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    #trex/[folder to save to]/[optional: starting name of all saved models (otherwise just epoch and iteration)]
    parser.add_argument('--reward_model_path', default='trex/reward_model_chaser/test1', help="name and location for learned model params, e.g. ./learned_models/breakout.params")

    args = parser.parse_args()

    # TODO (max): change these so they make sense
    # e.g. use some form of l1 regularization, add l2 reg, etc.
    # use more than 5 epochs (e.g. train until convergence)
    # there are other things to consider later, e.g. pytorch schedulers
    # (that change the learning rate over time)
    args.lr = 0.00005
    args.weight_decay = 0.0
    args.num_iter = 500 #maximum num times through training data 
    args.lam_l1=0.0 
    args.converg = .001
    args.stochastic = True

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args


def main():

    args = parse_config()
    # TODO (liang): add logging to this based on helpers/utils.py/log_this

    # TODO (max): make seeds work properly
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    print("Training reward for", args.env_name)
    
    Procgen_fn = lambda: ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    venv_fn = lambda: VecExtractDictObs(Procgen_fn(), "rgb")
    
    # here is where the T-REX procedure begins

    # collect a bunch of demonstrations from trained models

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    policy_model = ppo2.learn(env=venv_fn(), network=conv_fn, total_timesteps=0)
    demonstrations, learning_returns, learning_rewards = generate_procgen_demonstrations(venv_fn,
     policy_model, args.models_dir, eps_per_model = args.num_envs)
    # TODO: why is args.num_envs used as a placeholder for eps_per_model?


    # sort the demonstrations according to ground truth reward to simulate ranked demos

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    
    assert len(demonstrations) == len(learning_returns), "demos and rews are not of equal lengths"
    print([a[0] for a in zip(learning_returns, demonstrations)])
    # TODO (anton): move some of this code to the creating of training data
    # might have to redo how some of this works
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)

    num_snippets = args.num_snippets
    min_snippet_length = 10 #min length of trajectory for training comparison
    max_snippet_length = 100
    
    # TODO (anton): this process might be different depending on what the true reward model looks like
    # e.g. it's not very nice in coinrun

    training_obs, training_labels = create_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))

   
    # train a reward network using the demonstrations collected earlier and save it

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = RewardTrainer(args, device)

    trainer.learn_reward(training_obs, training_labels)
    trainer.save_model()
    
    # print out predicted cumulative returns and actual returns

    with torch.no_grad():
        pred_returns = [trainer.predict_traj_return(traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", trainer.calc_accuracy(training_obs, training_labels))


if __name__=="__main__":
    main()
