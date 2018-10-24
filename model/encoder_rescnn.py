# copied from previous version
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim


def res_block(input, kernal, filters, stride=1, clip_value=20):
    net = slim.conv2d(input, filters, kernal, stride)
    net = tf.minimum(net, clip_value)
    net = slim.conv2d(net, filters, kernal, stride, activation_fn=None)
    net = tf.add(net, input)
    net = tf.nn.relu(net)
    net = tf.minimum(net, clip_value)
    return net


def encoder(inputs,
            params,
            is_training=True,
            ):
    embedding_size = params['embedding_size']
    input_feature_dim = params['input_feature_dim']
    l2_regularization_weight = params['l2_regularization_weight']
    # adhoc fix, append the C dimension
    inputs = tf.expand_dims(inputs, 3)

    output = inputs
    with tf.variable_scope("ResCnnNet"):
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        }
        # set default parameters for slim layers: conv2d, fully_connected, batch_norm and dropout
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(
                                l2_regularization_weight) if l2_regularization_weight > 0 else None
                            ):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):

                filters = [64, 128, 256, 512]

                for i, filter in enumerate(filters):
                    output = slim.conv2d(output, filter, kernel_size=5, stride=2, activation_fn=None)
                    for n in range(3):
                        output = res_block(output, 3, filter)

                # NHWC format
                shape = tf.shape(output)
                # bs, ts,w,c -> bs,ts, w*c
                # input_feature_dim / 16
                output_feature_dim = math.ceil(input_feature_dim / 16) * filters[-1]  # 2048

                output = tf.reshape(output, (shape[0], shape[1], output_feature_dim))

                output = tf.reduce_mean(output, axis=1)

                output = slim.fully_connected(output, embedding_size, activation_fn=None)

                output = tf.nn.l2_normalize(output, 1, name="embedding")

                return output
