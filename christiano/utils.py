import os, time, datetime
import pickle
import json
from stable_baselines import PPO2
import argparse

def timeitt(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('time spent by %r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed 

def save_state(run_dir, i, reward_model, policy, data_buffer):

    save_dir =os.path.join(run_dir, "saved_states", str(i))
    os.makedirs(save_dir, exist_ok=True)

    policy_save_path = os.path.join(save_dir, 'policy')
    rm_save_path = os.path.join(save_dir, 'rm.pth')
    data_buff_save_path = os.path.join(save_dir, 'data_buff.pth')

    with open(rm_save_path, 'wb') as f:
        pickle.dump(reward_model, f)

    with open(data_buff_save_path, 'wb') as f:
        pickle.dump(data_buffer, f)  

    policy.save(policy_save_path)

def load_state(run_dir):

    state_dir = os.path.join(run_dir, "saved_states")
    i = max([int(f.name) for f in os.scandir(state_dir) if f.is_dir()])
    load_dir =os.path.join(state_dir, str(i))

    policy_load_path = os.path.join(load_dir, 'policy')
    rm_load_path = os.path.join(load_dir, 'rm.pth')
    data_buff_load_path = os.path.join(load_dir, 'data_buff.pth')

    args_path = os.path.join(run_dir, "config.json")
    with open(args_path) as f:
        args = argparse.Namespace()
        args.__dict__.update(json.load(f))

    reward_model = pickle.load(open(rm_load_path, 'rb'))
    data_buffer = pickle.load(open(data_buff_load_path, 'rb'))
    policy = PPO2.load(load_path = policy_load_path, **args.ppo_kwargs)

    return reward_model, policy, data_buffer, i+1


def setup_logging(args):

    #Setting up directory for logs
    if not args.log_name:
        args.log_name = args.env_name + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%-M_%S')

    run_dir = os.path.join(args.log_dir, args.log_name)
    os.makedirs(run_dir, exist_ok=True)
    monitor_dir = os.path.join(run_dir ,'EnvMonitor', 'monitor')
    os.makedirs(monitor_dir, exist_ok=True)
    video_dir = os.path.join(run_dir ,'video')
    os.makedirs(video_dir, exist_ok=True)

    print('\n=== Logging ===', flush=True)
    print(f'Logging to {run_dir}', flush=True)

    return run_dir, monitor_dir, video_dir

def store_args(args, run_dir):
    args_path = os.path.join(run_dir, 'config.json')
    with open(args_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
        print(f'Config file saved to: {args_path}', flush=True)

def load_args(args):
    run_dir = os.path.join(args.log_dir, args.log_name)
    args_path = os.path.join(run_dir, "config.json")
    with open(args_path) as f:
        args = argparse.Namespace()
        args.__dict__.update(json.load(f))

    args.resume_training = True
    
    return args