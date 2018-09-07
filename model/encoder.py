import tensorflow as tf
from model.attention import attention

def encoder(inputs,
             num_filters,
             blocks=3,
             kernel_size=3,
             use_batch_norm=True,
             is_training=True,
             pool_size=2,
             pool_strides=2,
             embedding_size=128):
    # inputs: batch_size, time steps, channel
    output = inputs
    with tf.variable_scope("encoder"):
        for l in range(blocks):
            with tf.variable_scope('block_{}'.format(l + 1)):
                output = tf.layers.conv1d(output, filters=num_filters, kernel_size=kernel_size, padding='same')
                if use_batch_norm:
                    output = tf.layers.batch_normalization(output, training=is_training)
                output = tf.nn.relu(output)
                output = tf.layers.max_pooling1d(output, pool_size=pool_size, strides=pool_strides)

        # to one fixed length: batch_size, num_channels, by using the attention mechanism
        output = attention(output)

        with tf.variable_scope('output_transformer'):
            output = tf.layers.dense(output, embedding_size)

    return output