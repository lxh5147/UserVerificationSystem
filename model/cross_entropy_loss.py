import tensorflow as tf


def cross_entropy_loss(labels, embeddings, num_classes):
    # labels: shape batch_size
    # embeddings: shape batch_size, dim
    logits = tf.layers.dense(inputs=embeddings, units=num_classes)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
