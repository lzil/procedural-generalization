import os

import tensorflow as tf

import yaml
import logging
import time
import json
import csv
import pandas as pd


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
    run_id = str(int(time.time() * 100))[-7:]
    print(f'Run id: {run_id} with name {log_name}')

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


# extract id from the path. a bit hacky but should get the job done
def get_id(path):
    rm_id = '.'.join(os.path.basename(path).split('.')[:-1])
    return rm_id


# helper function for filtering rows in a csv
def retain_row(row, constraints):
    for k,v in constraints.items():
        # respect maximum return constraints
        if 'demo_max_return' in constraints:
            if float(row['return']) > float(constraints['demo_max_return']):
                return False
        if 'demo_max_len' in constraints:
            if float(row['length']) > float(constraints['demo_max_len']):
                return False
        if 'demo_min_len' in constraints:
            if float(row['length']) < float(constraints['demo_max_len']):
                return False
        if 'rm_max_return' in constraints:
            if float(row['max_return']) > float(constraints['rm_max_return']):
                return False

        # all other constraints
        if row[k] != v:
            return False
    return True

# helper function to get the rows in a csv that matter
def filter_csv(path, constraints, max_rows=1000000):
    with open(path) as master:
        reader = csv.DictReader(master, delimiter=',')
        rows = []
        for row in reader:
            if not retain_row(row, constraints):
                continue
            rows.append(row)
            if len(rows) > max_rows:
                break

    return rows

# helper function using pandas
def filter_csv_pandas(path, constraints):
    infos = pd.read_csv(path)
    if 'env_name' in constraints:
        infos = infos[infos['env_name'] == constraints['env_name']]
    if 'mode' in constraints:
        infos = infos[infos['mode'] == constraints['mode']]
    if 'sequential' in constraints:
        infos = infos[infos['sequential'] == constraints['sequential']]
    if 'set_name' in constraints:
        infos = infos[infos['set_name'] == constraints['set_name']]
    if 'demo_min_len' in constraints:
        # using > here instead of >=
        infos = infos[infos['length'] > constraints['demo_min_len']]
    return infos



# https://stackoverflow.com/questions/19932130/iterate-through-folders-then-subfolders-and-print-filenames-with-path-to-text-f
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

