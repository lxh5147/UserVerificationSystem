import tensorflow as tf

from model.attention import attention


def encoder(inputs,
            params,
            is_training=True,
            ):
    # inputs: batch_size, time steps, channel
    filters = list(params['filters'])[0]
    blocks = params['blocks']
    kernel_size = params['kernel_size']
    is_training = is_training
    strides = params['strides']
    embedding_size = params['embedding_size']

    output = inputs
    with tf.variable_scope("encoder"):
        for l in range(blocks):
            with tf.variable_scope('block_{}'.format(l + 1)):
                output = tf.layers.conv1d(output,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          padding='same')
                output = tf.layers.batch_normalization(output,
                                                       training=is_training)
                output = tf.nn.relu(output)
                output = tf.layers.max_pooling1d(output,
                                                 pool_size=kernel_size,
                                                 strides=strides,
                                                 padding='same')

        # to one fixed length: batch_size, num_channels, by using the attention mechanism
        output = attention(output)
        with tf.variable_scope('output_transformer'):
            output = tf.layers.dense(output, embedding_size)

    return output
