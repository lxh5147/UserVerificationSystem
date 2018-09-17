import tensorflow as tf

def attention(inputs):
    '''
    A weighted average of the inputs, where the weight of an input corresponds to the attention paied on this input
    :param inputs: 3D float tensor, (batch, time_steps, dim)
    :return: weighted input, 2D float tensor, (batch, dim)
    '''
    # input: batch_size, time_steps, dim
    # output: batch_size, dim
    with tf.variable_scope("attention"):
        w = tf.get_variable("hidden", initializer=tf.zeros_initializer(), shape=inputs.shape[-1:])
        # batch_size, time_steps
        logits = tf.tensordot(w, tf.nn.tanh(inputs), axes=[0, 2])
        p = tf.nn.softmax(logits)
        # batch_size, time_steps,1
        p = tf.expand_dims(p, -1)
        # batch_size, dim
        # p*inputs element wise production
        a = tf.reduce_sum(p * inputs, axis=1)
        return a
