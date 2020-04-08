import os

import tensorflow as tf

import yaml
import logging
import time
import json


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)


# use yaml config files; note what is actually set via the config file
def add_yaml_args(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        # all(map(dic.pop, config))
        for c, v in config.items():
            dic[c] = v
            if c in dic.keys():
                logging.info(f'{c} is set via config: {v}')
            else:
                logging.warning(f'{c} is not set to begin with: {v}')
    return args

# produce run id and create log directory
def log_this(config, log_dir, log_name=None):
    run_id = str(int(time.time()))[4:]
    print(f'Run id: {run_id}')

    if log_name is None or len(log_name) == 0:
        log_name = run_id
    run_dir = os.path.join(log_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Logging to {run_dir}')
    # might want to send stdout here later too
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}')

    return run_dir, checkpoint_dir, run_id
