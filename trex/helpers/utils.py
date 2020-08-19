import os
import glob
import pickle 
import tensorflow as tf
import numpy as np
from shutil import copy2

import yaml
import time
import json
import csv

from scipy.stats import pearsonr
from scipy.stats import spearmanr

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
def log_this(config, log_dir, rm_id=''):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    print('\n=== Logging ===', flush=True)
    print(f'Run id: {run_id}', flush=True)
    run_dir = os.path.join(log_dir, rm_id, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Logging to {run_dir}', flush=True)
    config.run_dir = run_dir

    path_config = os.path.join(run_dir, 'config.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
    print('===============\n', flush=True)

    return run_dir


# extract id from the path. a bit hacky but should get the job done
# just returns the id if the id is the input
def get_id(path):
    if '.' not in path and '/' not in path:
        return path
    rm_id = '.'.join(os.path.basename(path).split('.')[:-1])
    return rm_id

# helper function using pandas
def filter_csv_pandas(infos, constraints):
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


def timeitt(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('time spent by %r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def store_model(state_dict_path, max_return, max_length, accs, args):

    csv_name = 'rm_infos.csv' if args.save_name is None else f'rm_infos_{args.save_name}.csv'
    info_path = os.path.join(args.save_dir, csv_name)

    if not os.path.exists(info_path):
        with open(info_path, 'w') as f:
            rew_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            rew_writer.writerow(['rm_id', 'method', 'env_name', 'mode',
                                 'num_dems', 'max_return', 'max_length',
                                 'sequential', 'train_acc', 'val_acc',
                                 'test_acc', 'pearson', 'spearman'])

    files_name = 'model_files' if args.save_name is None else f'model_files_{args.save_name}'
    model_dir = os.path.join(args.save_dir, files_name)
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, args.rm_id + '.rm')
    copy2(state_dict_path, save_path)

    train_acc, val_acc, test_acc, pearson, spearman = accs
    with open(info_path, 'a') as f:
        rew_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rew_writer.writerow([args.rm_id, 'trex', args.env_name, args.distribution_mode,
                            args.num_dems, max_return, max_length, args.sequential,
                            train_acc, val_acc, test_acc, pearson, spearman])


def get_demo(demo_id):
    # searches for the file with the given name in all subfolders,
    # then loads it and returns
    path = glob.glob('./**/' + demo_id + '.demo', recursive=True)[0]
    demo = pickle.load(open(path, 'rb'))

    return demo


def get_corr_with_ground(demos, net, verbose=False, baseline_reward=False):
    rs = []
    for dem in demos:
        if baseline_reward:
            r_prediction = len(dem['observations'])
        else:
            r_prediction = np.sum(net.predict_batch_rewards(dem['observations']))

        r_true = dem['return']

        rs.append((r_true, r_prediction))

    # calculate correlations and print them
    rs_by_var = list(zip(*rs))
    pearson_r, pearson_p = pearsonr(rs_by_var[0], rs_by_var[1])
    spearman_r, spearman_p = spearmanr(rs)

    if verbose:
        print(f'(pearson_r, spearman_r): {(pearson_r, spearman_r)}')

    return (pearson_r, spearman_r)
