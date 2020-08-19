import os
import sys
import glob

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.ppo2 import ppo2

from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

from helpers.ProxyRewardWrapper import ProxyRewardWrapper
from helpers.utils import add_yaml_args, log_this

from train_reward import RewardNet


def parse_config():
    parser = argparse.ArgumentParser(description='Procgen training, with a revised reward model')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='starpilot')
    parser.add_argument('--distribution_mode', type=str, default='easy',
                        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='LOGS/POLICY_LOGS')
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--rm_id', default='', type=str, help="reward model id, e.g. 109_8714")

    # logs every num_envs * nsteps
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=20)

    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--ent_coef', type=float, default=.01)
    parser.add_argument('--gamma', type=float, default=.999)
    parser.add_argument('--lam', type=float, default=.95)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--clip_range', type=float, default=.2)
    # this should be num_envs * nsteps * whatever
    # 65536 * 3000 <=~ 50_000_000
    parser.add_argument('--timesteps_per_proc', type=int, default=25_000_000) 
    parser.add_argument('--use_vf_clipping', action='store_true', default=True)


    args = parser.parse_args()

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args


def main():

    args = parse_config()
    run_dir = log_this(args, args.log_dir, args.rm_id)

    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=run_dir, format_strs=format_strs)

    logger.info("creating environment")

    venv = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

    if args.rm_id:
        # load pretrained network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = RewardNet().to(device)
        rm_path = glob.glob('./**/'+ args.rm_id + '.rm', recursive=True)[0]
        net.load_state_dict(torch.load(rm_path, map_location=torch.device(device)))

        # use batch reward prediction function instead of the ground truth reward function
        rew_func = lambda x: net.predict_batch_rewards(x)
        venv = ProxyRewardWrapper(venv, rew_func)
    else:
        # true environment rewards will be used
        pass

    venv = VecNormalize(venv=venv, ob=False, use_tf=False)

    # do the rest of the training as normal
    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)

    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    
    model = ppo2.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=args.timesteps_per_proc,
        save_interval=args.save_interval,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
        lam=args.lam,
        gamma=args.gamma,
        noptepochs=args.ppo_epochs,
        log_interval=args.log_interval,
        ent_coef=args.ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=args.use_vf_clipping,
        comm=comm,
        lr=args.learning_rate,
        cliprange=args.clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        load_path = args.load_path,
    )

    model.save(os.path.join(checkpoint_dir, 'final_model.parameters'))



if __name__ == '__main__':
    main()

