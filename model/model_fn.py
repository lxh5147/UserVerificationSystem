"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def _attention(inputs):
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
        # p*w element wise production
        a = tf.reduce_sum(p * w, axis=1)
        return a


def _encoder(inputs,
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
        output = _attention(output)

        with tf.variable_scope('output_transformer'):
            output = tf.layers.dense(output, embedding_size)

    return output


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator
    Args:
        features: input batch
        labels: labels of the inputs
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    inputs = features
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model

        embeddings = _encoder(inputs,
                              num_filters=params['num_filters'],
                              blocks=params['blocks'],
                              kernel_size=params['kernel_size'],
                              use_batch_norm=params['use_batch_norm'],
                              is_training=is_training,
                              pool_size=params['pool_size'],
                              pool_strides=params['pool_strides'],
                              embedding_size=params['embedding_size'])

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params['triplet_strategy'] == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params['margin'],
                                                squared=params['squared'])
    elif params['triplet_strategy'] == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params['margin'],
                                       squared=params['squared'])
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params['triplet_strategy'] == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params['triplet_strategy'] == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    global_step = tf.train.get_global_step()
    if params['use_batch_norm']:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
