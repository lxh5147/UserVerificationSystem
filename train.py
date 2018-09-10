import tensorflow as tf
from model.voice_dataset import dataset as get_dataset
from model.model_fn import model_fn
import argparse
import sys
import os

FLAGS = None


def _get_labels(labels_file):
    with open(labels_file) as f:
        lines = f.read().splitlines()
    # map a line to an ID
    ids = {}
    label_ids = []
    for line in lines:
        if line in ids:
            cur_id = ids[line]
            label_ids.append(cur_id)
        else:
            cur_id = len(ids)
            ids[line] = cur_id
            label_ids.append(cur_id)
    return label_ids, ids


def _get_wav_files(directory):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            files.append(os.path.join(r, file))
    return files


def _input_fn(wav_files,
              labels,
              batch_size,
              desired_samples,
              window_size_samples,
              window_stride_samples,
              magnitude_squared=True,
              dct_coefficient_count=40,
              is_training=True,
              buffer_size=1000):
    dataset = get_dataset(wav_files,
                          labels,
                          desired_samples,
                          window_size_samples,
                          window_stride_samples,
                          magnitude_squared,
                          dct_coefficient_count
                          )

    # Shuffle, repeat, and batch the examples.
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat().batch(batch_size)

    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def _create_model(model_dir=None,
                  config=None,
                  params=None):
    return tf.estimator.Estimator(model_fn,
                                  model_dir=model_dir,
                                  config=config,
                                  params=params,
                                  )


def _from_ms_to_samples(sample_rate, duration_ms):
    return int(sample_rate * duration_ms / 1000)


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the input function for training
    train_wav_files = _get_wav_files(os.path.join(FLAGS.data_dir, 'train'))
    train_labels, train_label_ids = _get_labels(os.path.join(FLAGS.data_dir, 'train_labels'))

    train_num_classes = len(train_label_ids)
    filters = map(lambda _: int(_), FLAGS.filters.split(','))
    model = _create_model(
        model_dir=FLAGS.model_dir,
        params={
            'filters': filters,
            'blocks': FLAGS.blocks,
            'kernel_size': FLAGS.kernel_size,
            'strides': FLAGS.strides,
            'embedding_size': FLAGS.embedding_size,
            'triplet_strategy': FLAGS.triplet_strategy,
            'margin': FLAGS.margin,
            'squared': FLAGS.squared,
            'learning_rate': FLAGS.learning_rate,
            'learning_rate_decay_rate': FLAGS.learning_rate_decay_rate,
            'learning_rate_decay_steps': FLAGS.learning_rate_decay_steps,
            'l2_regularization_weight': FLAGS.l2_regularization_weight,
            'triplet_loss_weight': FLAGS.triplet_loss_weight,
            'cross_entropy_loss_weight': FLAGS.cross_entropy_loss_weight,
            'num_classes': train_num_classes,
            'encoder': FLAGS.encoder
        })

    desired_samples = _from_ms_to_samples(FLAGS.sample_rate, FLAGS.desired_ms)
    window_size_samples = _from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_size_ms)
    window_stride_samples = _from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_stride_ms)
    train_input_fn = lambda: _input_fn(
        wav_files=train_wav_files,
        labels=train_labels,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=FLAGS.magnitude_squared,
        dct_coefficient_count=FLAGS.dct_coefficient_count,
        batch_size=FLAGS.batch_size
    )

    # model Model
    model.train(train_input_fn,
                steps=FLAGS.num_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./tmp_model',
        help='model_dir')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='model_dir')
    parser.add_argument(
        '--filters',
        type=str,
        default='64,128,256,512',
        help='filters')
    parser.add_argument(
        '--blocks',
        type=int,
        default=3,
        help='blocks')
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
        help='kernel_size')
    parser.add_argument(
        '--strides',
        type=int,
        default=2,
        help='strides of conv')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='embedding_size')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Sample rate of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.', )
    parser.add_argument(
        '--desired_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.')
    parser.add_argument(
        '--magnitude_squared',
        type=bool,
        default=True,
        help='magnitude_squared')
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='dct_coefficient_count')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch_size')
    parser.add_argument(
        '--triplet_strategy',
        type=str,
        default='batch_hard',
        help='triplet_strategy')
    parser.add_argument(
        '--margin',
        type=float,
        default=0.2,
        help='margin')
    parser.add_argument(
        '--squared',
        type=bool,
        default=True,
        help='squared')
    parser.add_argument(
        '--num_steps',
        type=int,
        default=10000,
        help='num_steps')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='learning_rate')
    parser.add_argument(
        '--learning_rate_decay_rate',
        type=float,
        default=0.96,
        help='Learning rate decay rate.')
    parser.add_argument(
        '--learning_rate_decay_steps',
        type=int,
        default=1000,
        help='Decay the learning rate after every those steps.')
    parser.add_argument(
        '--l2_regularization_weight',
        type=float,
        default=0.00001,
        help='Weight of L2 regularization.')
    parser.add_argument(
        '--triplet_loss_weight',
        type=float,
        default=1.,
        help='Weight of triplet loss.')
    parser.add_argument(
        '--cross_entropy_loss_weight',
        type=float,
        default=1.,
        help='Weight of cross entropy loss.')
    parser.add_argument(
        '--encoder',
        type=str,
        default='cnn',
        help='Encoder that encodes a wav to a vector. Use cnn or resnet')

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + _)
