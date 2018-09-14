import tensorflow as tf

from model.attention import attention


def clipped_relu(inputs):
    return tf.minimum(tf.maximum(inputs, 0), 20)


def res_block(input_tensor, kernel_size, filters, stage, block, is_training=True):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)
    output = input_tensor

    output = tf.layers.conv1d(output,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='glorot_uniform',
                              name=conv_name_base + '_conv_a'
                              )
    output = tf.layers.batch_normalization(output,
                                           training=is_training,
                                           name=conv_name_base + '_conv_a_bn')
    output = clipped_relu(output)
    output = tf.layers.conv1d(output,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='glorot_uniform',
                              name=conv_name_base + '_conv_b'
                              )
    output = tf.layers.batch_normalization(output,
                                           training=is_training,
                                           name=conv_name_base + '_conv_b_bn')
    output = input_tensor + output
    output = clipped_relu(output)
    return output


def conv_and_res_block(input_tensor, kernel_size, filters, strides, stage, blocks=3, is_training=True):
    conv_name = 'conv{}-s'.format(filters)
    output = input_tensor
    output = tf.layers.conv1d(output,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding='same',
                              kernel_initializer='glorot_uniform',
                              name=conv_name,
                              )
    output = tf.layers.batch_normalization(output,
                                           training=is_training,
                                           name=conv_name + '_bn')
    output = clipped_relu(output)
    for i in range(blocks):
        output = res_block(output,
                           kernel_size=kernel_size,
                           filters=filters,
                           stage=stage,
                           block=i,
                           is_training=is_training)

    return output


def encoder(inputs,
            params,
            is_training=True,
            ):
    # inputs: batch_size, time steps, channel
    filters_list = list(params['filters'])
    blocks = params['blocks']
    kernel_size = params['kernel_size']
    is_training = is_training
    strides = params['strides']
    embedding_size = params['embedding_size']

    output = inputs

    with tf.variable_scope("encoder"):
        for stage, filters in enumerate(filters_list):
            output = conv_and_res_block(output,
                                        kernel_size=kernel_size,
                                        filters=filters,
                                        strides=strides,
                                        stage=stage + 1,
                                        blocks=blocks,
                                        is_training=is_training)

        # to one fixed length: batch_size, num_channels, by using the attention mechanism
        output = attention(output)

        with tf.variable_scope('output_transformer'):
            output = tf.layers.dense(output, embedding_size)

    return output
