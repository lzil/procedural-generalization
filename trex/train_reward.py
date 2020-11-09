# adapted heavily from
# https://github.com/hiwonjoon/ICML2019-TREX/blob/master/atari/LearnAtariReward.py

import numpy as np
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import pandas as pd

import os
import sys

import random
import argparse

import logging

from helpers.utils import get_demo, get_corr_with_ground, log_this,\
                         add_yaml_args, store_model, filter_csv_pandas

sys.path.append('../')


def create_dataset(dems, num_snippets, min_snippet_length, max_snippet_length,
                   verbose=True, use_snippet_rewards=False, use_clip_heuristic=True):
    """
    This function takes a set of demonstrations and produces
    a training set consisting of pairs of clips with assigned preferences
    """

    if verbose:
        logging.info(f' {len(dems)} demonstrations provided')
        logging.info(f"demo lengths : {[d['length'] for d in dems]}")
        logging.info(f"demo returns : {[d['return'] for d in dems]}")
        demo_lens = [d['length'] for d in dems]
        logging.info(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')
        assert min_snippet_length < min(demo_lens), "One of the trajectories is too short"

    data = []
    n_honest = 0

    while len(data) < num_snippets:

        # pick two random demos
        i1, i2 = np.random.choice(len(dems), 2, replace=False)
        d0, d1 = sorted([dems[i1], dems[i2]], key=lambda x: x['return'])
        if d0['return'] == d1['return']:
            continue

        # first adjust max stippet length such that we can pick
        # the later starting clip from the better trajectory
        cur_min_len = min(d0['length'], d1['length'])
        cur_max_snippet_len = min(cur_min_len, max_snippet_length)
        # randomly choose snippet length
        cur_len = np.random.randint(min_snippet_length, cur_max_snippet_len)

        if use_clip_heuristic:
            # pick tj snippet to be later than ti
            d0_start = np.random.randint(cur_min_len - cur_len + 1)
            d1_start = np.random.randint(d0_start, d1['length'] - cur_len + 1)
        else:
            # pick randomly
            d0_start = np.random.randint(d0['length'] - cur_len)
            d1_start = np.random.randint(d1['length'] - cur_len)

        clip0 = d0['observations'][d0_start : d0_start+cur_len]
        clip1 = d1['observations'][d1_start : d1_start+cur_len]

        clip0_rew = np.sum(d0['rewards'][d0_start : d0_start+cur_len])
        clip1_rew = np.sum(d1['rewards'][d1_start : d1_start+cur_len])

        if clip0_rew <= clip1_rew:
            n_honest += 1
        elif use_snippet_rewards:
            # swap if incorrectly labeled and using true snippet rewards
            clip0, clip1 = clip1, clip0

        data.append(([clip0, clip1], np.array([1])))

    logging.info(f'set length: {len(data)}')

    return np.array(data), n_honest/num_snippets

# actual reward learning network


class RewardNet(nn.Module):
    def __init__(self, output_abs=False):
        super().__init__()
        self.output_abs = output_abs

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.MaxPool2d(4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.MaxPool2d(4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(11*11*32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 16, 7, stride=3),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16, 16, 5, stride=2),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16, 16, 3, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16, 16, 3, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16*16, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 1)
        # )

    def predict_returns(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        if self.output_abs:
            r = torch.abs(self.model(x))
        else:
            r = self.model(x)
        all_reward = torch.sum(r)
        all_reward_abs = torch.sum(torch.abs(r))
        return all_reward, all_reward_abs

    def predict_batch_rewards(self, batch_obs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            # get into NCHW format
            x = torch.tensor(batch_obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            # compute forward pass of reward network (we parallelize across
            # frames so batch size is length of partial trajectory)
            if self.output_abs:
                r = torch.abs(self.model(x))
            else:
                r = self.model(x) 
            return r.cpu().numpy().flatten()

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        all_r_i, abs_r_i = self.predict_returns(traj_i)
        all_r_j, abs_r_j = self.predict_returns(traj_j)
        return torch.stack((all_r_i, all_r_j)), abs_r_i + abs_r_j

# trainer wrapper in order to make training the reward model a neat process


class RewardTrainer:
    def __init__(self, args, device):
        self.device = device
        self.net = RewardNet(output_abs=args.output_abs).to(device)
        self.best_model = copy.deepcopy(self.net.state_dict())
        self.args = args

    # Train the network
    def learn_reward(self, train_set, val_set, test_set, test_dems):
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)

        max_val_acc = 0
        eps_no_max = 0

        with open(self.args.train_log, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['n_train_samples', 'train_acc', 'train_loss',
                             'val_acc', 'val_loss', 'test_acc', 'test_loss',
                             'pearson', 'spearman'])

            for epoch in range(self.args.max_num_epochs):
                epoch_loss = 0
                reward_list = []
                abs_reward_list = []
                np.random.shuffle(train_set)
                # each epoch consists of some updates - NOT passing through whole test set.
                for i, ([traj_i, traj_j], label) in enumerate(train_set[:self.args.epoch_size]):

                    ti = torch.from_numpy(traj_i).float().to(self.device)
                    tj = torch.from_numpy(traj_j).float().to(self.device)
                    lb = torch.from_numpy(label).to(self.device)

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs, abs_rewards = self.net.forward(ti, tj)

                    reward_list.append(outputs[0].cpu().item())
                    reward_list.append(outputs[1].cpu().item())
                    abs_reward_list.append(abs_rewards.cpu().item())

                    outputs = outputs.unsqueeze(0)

                    # L1 regularization on the output
                    l1_reg = abs_rewards * self.args.lam_l1

                    loss = loss_criterion(outputs, lb) + l1_reg
                    loss.backward()
                    optimizer.step()

                    item_loss = loss.item()
                    epoch_loss += item_loss

                epoch_loss /= self.args.epoch_size
                train_acc, train_loss = self.calc_accuracy(train_set[:1000])
                # keep validation set to 1000
                val_acc, val_loss = self.calc_accuracy(val_set[:1000])
                test_acc, test_loss = self.calc_accuracy(test_set)
                # calculating correlations on the subset
                # of all test demos to save time
                pearson, spearman = get_corr_with_ground(test_dems[:100], self.net)

                writer.writerow([epoch*self.args.epoch_size, train_acc, train_loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item(), pearson, spearman])

                avg_reward = np.mean(np.array(reward_list))
                avg_abs_reward = np.mean(np.array(abs_reward_list))

                logging.info(f"n_samples: {(epoch+1)*self.args.epoch_size:6g} | loss: {epoch_loss:5.2f} | rewards mean/mean_abs: {avg_reward.item():5.2f}/{avg_abs_reward.item():.2f} | pc: {pearson:5.2f} | sc: {spearman:5.2f}")
                logging.info(f'   | train_acc : {train_acc:6.4f} | val_acc : {val_acc:6.4f} | test_acc : {test_acc:6.4f}')
                logging.info(f'   | train_loss: {train_loss:6.4f} | val_loss: {val_loss:6.4f} | test_loss: {test_loss:6.4f}')

                if val_acc > max_val_acc:
                    self.save_model()
                    max_val_acc = val_acc
                    eps_no_max = 0
                    accs = (train_acc, val_acc, test_acc, pearson, spearman)
                else:
                    eps_no_max += 1

                # Early stopping
                if eps_no_max >= self.args.patience:
                    logging.info(f'Early stopping after epoch {epoch}')
                    # loading the model with the best validation accuracy
                    self.net.load_state_dict(self.best_model)
                    logging.info('calculating correlations on all of the available test demos')
                    pearson, spearman = get_corr_with_ground(test_dems, self.net)
                    accs = (*accs[:3], pearson, spearman)
                    break

        logging.info("finished training")
        return os.path.join(self.args.run_dir, 'reward_best.pth'), accs

    # save the final learned model
    def save_model(self):
        torch.save(self.net.state_dict(), os.path.join(self.args.run_dir, 'reward_best.pth'))
        self.best_model = copy.deepcopy(self.net.state_dict())

    # calculate and return accuracy on entire training set
    def calc_accuracy(self, data):
        criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        total_loss = 0.

        with torch.no_grad():
            for [traj_i, traj_j], label in data:
                ti = torch.from_numpy(traj_i).float().to(self.device)
                tj = torch.from_numpy(traj_j).float().to(self.device)
                lb = torch.from_numpy(label).to(self.device)

                # forward to get logits
                rewards, abs_rewards = self.net(ti, tj)
                _, pred_label = torch.max(rewards, 0)
                if pred_label.item() == lb:
                    num_correct += 1.

                loss = criterion(rewards.unsqueeze(0), lb).cpu().item() + abs_rewards * self.args.lam_l1
                total_loss += loss

        return num_correct / len(data), total_loss / len(data)

    # purpose of these two functions is to get predicted return (via reward net) from the trajectory given as input
    # possibly get rid of these two functions and merge with the prediction functions in RewardNet
    def predict_reward_sequence(self, traj):
        rewards_from_obs = []
        with torch.no_grad():
            for s in traj:
                r = self.net.predict_returns(torch.from_numpy(np.array([s])).float().to(self.device))[0].item()
                rewards_from_obs.append(r)
        return rewards_from_obs

    def predict_traj_return(self, traj):
        return sum(self.predict_reward_sequence(traj))


def parse_config():
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy',
                        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--seed', type=int, help="random seed for experiments")
    parser.add_argument('--sequential', type=int, default=0,
                        help='0 means not sequential, any other number creates sequential env with start_level = args.sequential')

    parser.add_argument('--num_dems',type=int, default=12, help='Number off demonstrations to train on')
    parser.add_argument('--max_return',type=float, default=1.0,
                        help='Maximum return of the provided demonstrations as a fraction of max available return')
    parser.add_argument('--num_snippets', default=50000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--min_snippet_length', default=20, type=int, help="Min length of tracjectory for training comparison")
    parser.add_argument('--max_snippet_length', default=100, type=int, help="Max length of tracjectory for training comparison")

    parser.add_argument('--epoch_size', default=1000, type=int, help='How often to measure validation accuracy')
    parser.add_argument('--max_num_epochs', type=int, default=50, help='Number of epochs for reward learning')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--lr', type=float, default=5e-5, help='reward model learning rate')
    parser.add_argument('--lam_l1', type=float, default=0.001, help='l1 penalization of abs value of output')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay of updates')
    parser.add_argument('--output_abs', action='store_true', help='absolute value the output of reward model')
    parser.add_argument('--use_snippet_rewards', action='store_true', help='use true rewards instead of demonstration ones')
    parser.add_argument('--use_clip_heuristic', type=bool, default=True, help='always pick later part of a better trajectory when generating clips')

    parser.add_argument('--log_dir', default='LOGS/RM_LOGS', help='Training logs directory')

    parser.add_argument('--demo_csv', nargs='+', default=['demos/demo_infos.csv'], help='path to csv files with demo info')

    parser.add_argument('--save_dir', default='reward_models', help='where the trained models and csv get stored')
    parser.add_argument('--save_name', default=None, help='suffix to the name of the csv/file folder for saving')

    args = parser.parse_args()

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args


def main():

    args = parse_config()

    # do seed creation before log creation
    print('Setting up logging and seed creation', flush=True)
    if args.seed:
        seed = args.seed
    else:
        seed = random.randint(1e6, 1e7-1)
        args.seed = seed
    # Reward model id is derived from the seed
    args.rm_id = '_'.join([str(seed)[:3], str(seed)[3:]])

    run_dir = log_this(args, args.log_dir, args.rm_id)
    args.run_dir = run_dir

    args.log_path = os.path.join(run_dir, 'print_out.txt')
    args.train_log = os.path.join(run_dir, 'train_log.csv')

    logging.basicConfig(format='%(message)s', filename=args.log_path, level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    logging.info(f'Save name: {args.save_name}')
    logging.info(f'Reward model id: {args.rm_id}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    # here is where the T-REX procedure begins
    constraints = {
        'env_name': args.env_name,
        'mode': args.distribution_mode,
        'demo_min_len': args.min_snippet_length,
        'sequential': args.sequential
    }

    for path in args.demo_csv:
        all_rows = pd.read_csv(path)
        train_rows = filter_csv_pandas(all_rows, {'set_name': 'train', **constraints})
        test_rows = filter_csv_pandas(all_rows, {'set_name': 'test', **constraints})

    logging.info(f'Filtered demos: {len(train_rows)} training demos available, {args.num_dems} requested')

    logging.info('Creating training set ...')

    # implementing uniformish distribution of demo returns
    min_return = train_rows.min()['return']
    max_return = (train_rows.max()['return'] - min_return) * args.max_return + min_return

    rew_step = (max_return - min_return)/ 4
    dems = []
    seeds = []
    while len(dems) < args.num_dems:

        high = min_return + rew_step 
        while (high <= max_return) and (len(dems) < args.num_dems):
            # crerate boundaries to pick the demos from, and filter demos accordingly
            low = high - rew_step
            filtered_dems = train_rows[(train_rows['return'] >= low) & (train_rows['return'] <= high)]
            # make sure we have only unique demos
            new_seeds = filtered_dems[~filtered_dems['demo_id'].isin(seeds)]['demo_id']
            # choose random demo and append
            if len(new_seeds) > 0:
                chosen_seed = np.random.choice(new_seeds, 1).item()
                dems.append(get_demo(chosen_seed))
                seeds.append(chosen_seed)
            high += rew_step

    max_demo_return = max([demo['return'] for demo in dems])
    max_demo_length = max([demo['length'] for demo in dems])

    # make training and validation datasets separate
    # so there are two different calls to create the two datasets

    train_dems = dems[ : int(args.num_dems * 0.8)]
    val_dems = dems[int(args.num_dems * 0.8) : ]

    train_set, _ = create_dataset(
        dems=train_dems,
        num_snippets=args.num_snippets,
        min_snippet_length=args.min_snippet_length,
        max_snippet_length=args.max_snippet_length,
        verbose=False,
        use_snippet_rewards=args.use_snippet_rewards,
        use_clip_heuristic=args.use_clip_heuristic
        )

    logging.info('Creating validation set ...')

    val_set, _ = create_dataset(
        dems=val_dems,
        num_snippets=1000,
        min_snippet_length=args.min_snippet_length,
        max_snippet_length=args.max_snippet_length,
        verbose=False,
        use_snippet_rewards=args.use_snippet_rewards,
        use_clip_heuristic=args.use_clip_heuristic
        )

    # acquiring test demos for correlations and test accuracy
    logging.info('Creating test set ...')
    n_test_demos = 100
    test_dems = []
    for dem in test_rows['demo_id']:
        test_dems.append(get_demo(dem))

    test_set, true_test_acc = create_dataset(
        dems=test_dems[:n_test_demos],
        num_snippets=1000,
        min_snippet_length=args.min_snippet_length,
        max_snippet_length=args.max_snippet_length,
        verbose=False,
        use_snippet_rewards=args.use_snippet_rewards,
        use_clip_heuristic=args.use_clip_heuristic
    )

    logging.info(f'GT reward test set accuracy = {true_test_acc}')
    # train a reward network using the dems collected earlier and save it
    logging.info("Training reward model for %s ...", args.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = RewardTrainer(args, device)

    state_dict_path, accs = trainer.learn_reward(train_set, val_set, test_set, test_dems)

    # print out predicted cumulative returns and actual returns
    # merge this successfully with anton's branch to print test return examples
    with torch.no_grad():
        logging.info('_____TRAIN set_____')
        logging.info('true     |predicted')
        for demo in sorted(dems[:20], key=lambda x: x['return']):
            logging.info(f"{demo['return']:<9.2f}|{trainer.predict_traj_return(demo['observations']):>9.2f}")

    with torch.no_grad():
        logging.info('______TEST set_____')
        logging.info('true     |predicted')
        for demo in sorted(test_dems[:20], key=lambda x: x['return']):
            logging.info(f"{demo['return']:<9.2f}|{trainer.predict_traj_return(demo['observations']):>9.2f}") 

    logging.info(f"Final train set accuracy {trainer.calc_accuracy(train_set[:5000])[0]}")

    store_model(state_dict_path, max_demo_return, max_demo_length, accs, args)


if __name__ == "__main__":
    main()
