import os

import tensorflow as tf

import yaml
import logging
import time
import json


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# use yaml config files; note what is actually set via the config file
def add_yaml_args(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        # all(map(dic.pop, config))
        for c, v in config.items():
            dic[c] = v
            # if c in dic.keys():
            #     logging.info(f'{c} is set via config: {v}')
            # else:
            #     logging.warning(f'{c} is not set to begin with: {v}')
    return args

# produce run id and create log directory
def log_this(config, log_dir, log_name=None, checkpoints=True):
    run_id = str(int(time.time()))[4:]
    print(f'Run id: {run_id}')

    if log_name is None or len(log_name) == 0:
        log_name = run_id
    run_dir = os.path.join(log_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)

    if checkpoints:
        checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
        os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f'{run_id}.log')

    print(f'Logging to {run_dir}')
    # might want to send stdout here later too
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}')

    # major TODO to change this to be more widely usable and not specific to applications
    if checkpoints:
        # used for reward model training
        return log_path, checkpoint_dir, run_id
    else:
        # used for correlations
        return run_dir, run_id

import scipy
import numpy as np

def least_l2_affine(
    source: np.ndarray, target: np.ndarray, shift: bool = True, scale: bool = True
):
    """Finds the squared-error minimizing affine transform.

    Args:
        source: a 1D array consisting of the reward to transform.
        target: a 1D array consisting of the target to match.
        shift: affine includes constant shift.
        scale: affine includes rescale.

    Returns:
        (shift, scale) such that (scale * reward + shift) has minimal squared-error from target.

    Raises:
        ValueError if source or target are not 1D arrays, or if neither shift or scale are True.
    """
    if source.ndim != 1:
        raise ValueError("source must be vector.")
    if target.ndim != 1:
        raise ValueError("target must be vector.")
    if not (shift or scale):
        raise ValueError("At least one of shift and scale must be True.")

    a_vals = []
    if shift:
        # Positive and negative constant.
        # The shift will be the sum of the coefficients of these terms.
        a_vals += [np.ones_like(source), -np.ones_like(source)]
    if scale:
        a_vals += [source]
    a_vals = np.stack(a_vals, axis=1)
    # Find x such that a_vals.dot(x) has least-squared error from target, where x >= 0.
    coefs, _ = scipy.optimize.nnls(a_vals, target)

    shift_param = 0.0
    scale_idx = 0
    if shift:
        shift_param = coefs[0] - coefs[1]
        scale_idx = 2

    scale_param = 1.0
    if scale:
        scale_param = coefs[scale_idx]

    return shift_param, scale_param
