"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.cross_entropy_loss import cross_entropy_loss


def _get_encoder(encoder_name):
    assert encoder_name in ['cnn', 'resnet']
    if encoder_name == 'cnn':
        from model.encoder_cnn import encoder as encoder_cnn
        return encoder_cnn
    elif encoder_name == 'resnet':
        from model.encoder_resnet import encoder as encoder_resnet
        return encoder_resnet


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
    encoder = _get_encoder(params['encoder'])

    embeddings = encoder(inputs,
                         params=params,
                         is_training=is_training,
                         )
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params['triplet_strategy'] == "batch_all":
        loss_triplet, fraction = batch_all_triplet_loss(labels,
                                                        embeddings,
                                                        margin=params['margin'],
                                                        squared=params['squared'])
    elif params['triplet_strategy'] == "batch_hard":
        loss_triplet = batch_hard_triplet_loss(labels,
                                               embeddings,
                                               margin=params['margin'],
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
        return tf.estimator.EstimatorSpec(mode, loss=loss_triplet, eval_metric_ops=eval_metric_ops)

    # Build loss
    loss = 0

    # Apply triplet loss
    triplet_loss_weight = params['triplet_loss_weight']
    if triplet_loss_weight > 0:
        if params['triplet_strategy'] == "batch_all":
            tf.summary.scalar('fraction_positive_triplets', fraction)
        tf.summary.scalar('loss_triplet', loss_triplet)
        loss += triplet_loss_weight * loss_triplet

    # Apply cross entropy loss
    cross_entropy_loss_weight = params['cross_entropy_loss_weight']
    if cross_entropy_loss_weight > 0:
        loss_cross_entropy = cross_entropy_loss(labels=labels,
                                                embeddings=embeddings,
                                                num_classes=params['num_classes'])
        tf.summary.scalar('loss_cross_entropy', loss_cross_entropy)
        loss += cross_entropy_loss_weight * loss_cross_entropy

    # Finally, apply weight regularization
    l2_regularization_weight = params['l2_regularization_weight']
    if l2_regularization_weight > 0:
        loss_reg = l2_regularization_weight * tf.add_n([tf.reduce_sum(tf.square(w)) for w in tf.trainable_variables()])
        tf.summary.scalar('loss_reg', loss_reg)
        loss += loss_reg

    # Define training step that minimizes the loss with the Adam optimizer
    global_step = tf.train.get_global_step()
    lr_decay_rate = params['learning_rate_decay_rate']
    lr_decay_steps = params['learning_rate_decay_steps']
    lr_start = params['learning_rate']
    learning_rate = tf.train.exponential_decay(learning_rate=lr_start,
                                               global_step=global_step,
                                               decay_rate=lr_decay_rate,
                                               decay_steps=lr_decay_steps)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Add a dependency to update the moving mean and variance for batch normalization
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
