import tensorflow as tf


def cross_entropy_loss(labels, embeddings, num_classes):
    '''
    Compute the cross entropy loss
    :param labels: 1D integer tensor, (batch_size), the label of each embedding
    :param embeddings: 2D float tensor, (batch_size, dim)
    :param num_classes: int, the number of total labels
    :return: float scalar, the cross entropy loss
    '''
    # labels: shape batch_size
    # embeddings: shape batch_size, dim
    logits = tf.layers.dense(inputs=embeddings, units=num_classes)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
