from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


class ImpalaPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(ImpalaPolicy, self).__init__(*args, **kwargs,
                                           cnn_extractor = build_impala_cnn,
                                           feature_extraction="cnn")




# # Default CNN policy used for training
# class DefaultPolicy(CnnPolicy):
#     '''
#     Here we define default policy used for training an agent
#     Guide on how to do int: https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy
#     '''
#     def __init__(self, *args, **kwargs):
#         # super(DefaultPolicy, self).__init__(*args, **kwargs,
#         #                                    net_arch=[dict(pi=[128, 128, 128],
#         #                                                   vf=[128, 128, 128])],
#         #                                    feature_extraction="mlp")


# ## define more

# register_policy('DefaultPolicy', DefaultPolicy)
