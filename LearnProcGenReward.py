import numpy as np
import torch
import torch.nn as nn

def procgen_generate_demos(env_fn, model_dir, model, eps_per_model):

    eps = []
    for model_file in os.scandir(model_dir):
        model.load(model_file)
        collector = ProcGenRunner(env_fn, model, 512)
        eps.extend(collector.collect_episodes(eps_per_model))


    demonstrations = [episode['observations'] for episode in eps]
    learning_returns = [episode['return'] for episode in eps]
    learning_rewards = [episode['rewards'] for episode in eps]


    return demonstrations, learning_returns, learning_rewards


# TODO: could use a DataLoader for this entire process??
def create_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    #demonstrations should be sorted by increasing returns
    max_traj_len = 0
    training_obs = []
    training_labels = []

    n_demos = len(demonstrations)
    demo_lens = [len(t) for t in demonstrations]
    print(f'demo length: min = {min(demo_lens)}, max = {max(demo_lens)}')
    #fixed size snippets with progress prior
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
        # TODO: why?? is it so that tj will be more representative of a successful trajectory?
        ti_start = np.random.randint(cur_min_len - cur_len + 1)
        tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - cur_len + 1)

        # TODO: not sure what the comment in the below line means
        traj_i = demonstrations[ti][ti_start:ti_start+cur_len] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+cur_len]

        # TODO: traj_i and traj_j should have the same lengths, right?
        max_traj_len = max(max_traj_len, len(traj_i), len(traj_j))


        # TODO: why is this label random???? is this why tj snippet needs to be later than ti? still don't understand
        #randomize label
        label = np.random.randint(2)
        if label:
            training_obs.append((traj_i, traj_j))
        else:
            training_obs.append((traj_j, traj_i))
        training_labels.append(label)

    print(f"max traj length: {max_traj_len}")
    return training_obs, training_labels


class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
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

    def cum_return(self, traj):
        # TODO: why not just use predict_rewards, and then return the sum and absolute value sum of that?
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        with torch.no_grad():
            r = self.net(x)
        sum_rewards = torch.sum(r)
        sum_abs_rewards = torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def predict_rewards(self, batch_obs):
        with torch.no_grad():
            x = torch.tensor(batch_obs, dtype=torch.float32).permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
            r = self.net(x)
            return r.numpy().flatten()

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        # TODO: could you use torch.stack here instead of cat?
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j



# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        # TODO: could probably use a DataLoader for all of this??
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            # TODO: consider l2 regularization? why just l1
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 1000 == 999:
                #print(i)
                print("epoch {}, step {}: loss {}".format(epoch,i, cum_loss))
                print(f'absolute rewards = {abs_rewards.item()}')
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        # TODO: again, should be able to use a DataLoader instead of going through all this computation?
        for i in range(len(training_inputs)):
            # again, could just use a DataLoader
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)



# purpose of these two functions is to get predicted return (via reward net) from the trajectory given as input
def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":

    import time
    import copy
    import os

    import tensorflow as tf

    from baselines.ppo2 import ppo2
    from procgen import ProcgenEnv
    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    from baselines.common.vec_env import VecExtractDictObs
    from baselines.common.models import build_impala_cnn

    from traject_collector import ProcGenRunner

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import argparse
    parser = argparse.ArgumentParser(description='Default arguments to initialize and load the model and env')
    parser.add_argument('--env_name', type=str, default='chaser')
    parser.add_argument('--distribution_mode', type=str, default='hard',
        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_envs', type=int, default=2, help="number of demos per model")
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--num_snippets', default=6000, type=int,
        help="number of short subtrajectories to sample")
    parser.add_argument('--models_dir', default = "chaser_model_dir", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--reward_model_path', default='reward_model', help="name and location for learned model params, e.g. ./learned_models/breakout.params")

    args, unknown = parser.parse_known_args()


    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", args.env_name)
    num_snippets = args.num_snippets
    # TODO: why 20?
    min_snippet_length = 20 #min length of trajectory for training comparison
    max_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True

    venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    # TODO: what is the network actually learning here? not sure what's going on in this whole block
    model = ppo2.learn(env=venv, network=conv_fn, total_timesteps=0)

    env_fn = lambda: VecExtractDictObs(ProcgenEnv(num_envs=args.num_envs, env_name=args.env_name, num_levels=args.num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode),"rgb")

    #creates collector that would sample from env with model
    # TODO: are you using the collector here? or just in procgen_generate_demos?
    collector = ProcGenRunner(env_fn, model, 256)

    # TODO: why is args.num_envs used for eps_per_model
    demonstrations, learning_returns, learning_rewards = procgen_generate_demos(env_fn, args.models_dir, model, args.num_envs)  
    #sort the demonstrations according to ground truth reward to simulate ranked demos


    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    
    assert len(demonstrations) == len(learning_returns), "demos and rews are not of equal lengths"
    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    
    training_obs, training_labels = create_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
   
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path)
    #save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

