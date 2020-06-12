class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips"""
    def __init__(self, max_size):
        pass


    def add_data(self, data):
        pass

    def sample_batch(self, n):
        pass

    def sample_validation_batch(self, n):
        pass

def train_reward(reward_model, data_buffer, num_batches):
    '''
    Traines a given reward_model for num_batches from data_buffer
    Returns new reward_model
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
    pass

