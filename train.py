from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

# custom imports to make documenting easier
import yaml
import json
import logging
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from helpers.utils import add_yaml_args, log_this


# figure out the right configuration for the run
def parse_config():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_name', type=str, default='')
    # logs every num_envs * nsteps
    parser.add_argument('--log_interval', type=int, default=20)
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
    # this should be num_envs * nsteps
    # 65536 * 3000 <=~ 50_000_000
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000) 
    parser.add_argument('--use_vf_clipping', action='store_true', default=True)

    args = parser.parse_args()

    if args.config is not None:
        args = add_yaml_args(args, args.config)

    return args

def main():

    args = parse_config()

    # to log the results consistently
    LOG_DIR = os.path.join(args.log_dir, str(args.num_levels), args.env_name)
    run_dir,ckpt_dir, run_id = log_this(args, LOG_DIR, args.log_name)

    # mpi work
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
    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)


    #ppo learns venv: procgenenv -> vecextractdictobs -> vecmonitor -> vecnormalize

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2.learn(
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
        load_path=args.load_path
    )

if __name__ == '__main__':
    main()
