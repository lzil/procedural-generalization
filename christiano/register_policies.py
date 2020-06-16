from stable_baselines.common.policies import CnnPolicy, register_policy


# Default CNN policy used for training
class DefaultPolicy(CnnPolicy):
    '''
    Here we define default policy used for training an agent
    Guide on how to do int: https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy
    '''
    def __init__(self, *args, **kwargs):
        # super(DefaultPolicy, self).__init__(*args, **kwargs,
        #                                    net_arch=[dict(pi=[128, 128, 128],
        #                                                   vf=[128, 128, 128])],
        #                                    feature_extraction="mlp")


## define moe

register_policy('DefaultPolicy', DefaultPolicy)
