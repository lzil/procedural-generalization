class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips
    
    Each entry is ([clip0, clip1], label)
    clip0, clip2 : lists of observations
    label : float in range [0,1] corresponding to which clip is preferred
    label can also be 0.5 meaning clips are equal, or e.g. 0.95 corresponding 
    to noize in labeling 
    """

    def __init__(self, max_size):
        pass


    def add(self, data):
        pass

    def sample_batch(self, n):
        #probably adding noize to observations as a regularization (Ibarz et al. page 15)
        pass

    def sample_validation_batch(self, n):
        pass

    def get_size(self):
        '''returns buffer size'''

class RewardNet(nn.Module):
    """Here we set up a callable reward model

    Should have batch normalizatoin and dropout on conv layers
    
    """


def train_reward(reward_model, data_buffer, num_batches):
    '''
    Traines a given reward_model for num_batches from data_buffer
    Returns new reward_model

    Must have:
        Adaptive L2-regularization based on train vs validation loss
        L2-loss on the output
        Output normalized to 0 mean and 0.05 variance across data_buffer
        (Ibarz et al. page 15)
        
    '''
    pass

def train_policy(env, reward_model, policy, num_steps):
    '''
    Creates new environment by wrapping the env, with ProxyRewardWrapper given the reward_model.
    Traines policy in the new envirionment for num_steps
    Returns retrained policy
    '''
    pass

def collect_annotations(env, policy, num_pairs, clip_size):
    '''Collects episodes using the provided policy, slices them to snippets of given length,
    selects pairs randomly and annotates 
    Returns a list of tuples ([clip0, clip1], label), where label is float in [0,1]

    '''
    env.set_maxsteps(clip_size * 2 * num_pairs+10)
    clip_pool = []

    obs = env.reset()
    while len(clip_pool) < num_pairs *2:
        clip = {}
        clip['observations'] = []
        clip['return'] = 0
        while len(clip) < clip_size:
            # _states are only useful when using LSTM policies
            action, _states = policy.predict(obs)
            clip['observations'].append(obs)
            obs, reward, done, info = env.step(action)    
            clip['return'] += reward

        clip_pool.append(clip)

    clip_pairs = np.random.choice(clip_pool, (num_pairs, 2), replace = False)
    data = []
    for clip0, clip1 in clip_pairs:
        if clip0['return'] > clip1['return']:
            label = 0
        elif clip0['return'] < clip1['return']:
            label = 1 
        elif clip0['return'] == clip1['return']:
            label = 0.5

        data.append(([clip0['observations'], clip1['observations']], label))


def main():
    ##setup args
    args.init_buffer_size = 500
    args.clip_size = 25
    args.env_name = 'fruitbot'
    args.steps_per_iter = 10**5
    args.pairs_per_iter = 10**5
    args.pairs__in_batch = 16

    #initializing objects
    policy = PPO2(MlpPolicy, env, verbose=1)
    env = gym_procgen_continuous(env_name = args.env_name)
    reward_model = RewardNet()
    data_buffer = AnnotationBuffer()


    initial_data = collect_annotations(env, policy, args.init_buffer_size, args.clip_size)
    data_buffer.add(initial_data)

    num_batches = int(args.pairs_per_iter / args.pairs_in_batch)

    for i in args.num_iters:
        num_pairs = get_num_pairs()
        policy_save_path = 
        rm_save_path = 

        reward_model = train_reward(reward_model, data_buffer, num_batches) 
        policy = train_policy(env, reward_model, policy, args.steps_per_iter)
        annotations = collect_annotations(env, policy, num_pairs, args.clip_size)
        data_buffer.add(annotations)
        
        reward_model.save(rm_save_path)
        policy.save(policy_save_path)
