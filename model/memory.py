import tensorflow as tf


def read_memory(inputs, num_cells=10):
    '''
    A weighted average of the memory cells, where the weight of an memory cell corresponds
    to the attention on this cell
    :param inputs: 2D float tensor, (batch, dim)
    :return: weighted memory cell, 2D float tensor, (batch, dim)
    '''
    with tf.variable_scope("memory"):
        # num_cells, dim
        # TODO: other proper initializer
        cells = tf.get_variable("cells",
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                shape=tf.TensorShape([num_cells, inputs.shape[-1]]))
        # batch_size, num_cells
        logits = tf.tensordot(inputs, cells, axes=[1, 1])
        p = tf.nn.softmax(logits)
        # batch_size, dim
        outputs = tf.matmul(p, cells)
        return outputs
