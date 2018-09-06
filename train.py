import tensorflow as tf
from model.voice_dataset import dataset as get_dataset
from model.model_fn import model_fn
import argparse
import sys
import os

FLAGS = None

def get_labels(labels_file):
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
    return label_ids


def get_wav_files(directory):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            files.append(os.path.join(r, file))
    return files


def input_fn(wav_files,
             labels_file,
             batch_size,
             desired_samples,
             window_size_samples,
             window_stride_samples,
             desired_channels=1,
             magnitude_squared=True,
             dct_coefficient_count=40,
             is_training=True):
    labels = get_labels(labels_file)

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
        dataset = dataset.shuffle(buffer_size=1000).repeat().batch(batch_size)

    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def create_model(model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None):
    return tf.estimator.Estimator(model_fn,
                                  model_dir=model_dir,
                                  config=config,
                                  params=params,
                                  # warm_start_from=warm_start_from
                                  )


def from_ms_to_samples(sample_rate, duration_ms):
    return int(sample_rate * duration_ms / 1000)


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    model = create_model(
        model_dir=FLAGS.model_dir,
        # warm_start_from=FLAGS.warm_start_from,
        params={
            'num_filters': FLAGS.num_filters,
            'blocks': FLAGS.blocks,
            'kernel_size': FLAGS.kernel_size,
            'use_batch_norm': FLAGS.use_batch_norm,
            'pool_size': FLAGS.pool_size,
            'pool_strides': FLAGS.pool_strides,
            'embedding_size': FLAGS.embedding_size,
            'triplet_strategy': FLAGS.triplet_strategy,
            'margin': FLAGS.margin,
            'squared': FLAGS.squared,
            'learning_rate': FLAGS.learning_rate
        })

    # Define the input function for training
    train_wav_files = get_wav_files(os.path.join(FLAGS.data_dir, 'train'))
    train_labels_file = os.path.join(FLAGS.data_dir, 'train_labels')
    desired_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.desired_ms)
    window_size_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_size_ms)
    window_stride_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_stride_ms)
    train_input_fn = lambda: input_fn(
        wav_files=train_wav_files,
        labels_file=train_labels_file,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=FLAGS.magnitude_squared,
        dct_coefficient_count=FLAGS.dct_coefficient_count,
        batch_size=FLAGS.batch_size,
        desired_channels=FLAGS.desired_channels
    )

    # model Model
    model.train(train_input_fn, steps=FLAGS.num_steps)


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
    # parser.add_argument(
    #     '--warm_start_from',
    #     type=,
    #     default='',
    #     help='warm_start_from')
    parser.add_argument(
        '--num_filters',
        type=int,
        default=5,
        help='num_filters')
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
        '--use_batch_norm',
        type=bool,
        default=True,
        help='use_batch_norm')
    parser.add_argument(
        '--pool_size',
        type=int,
        default=2,
        help='pool_size')
    parser.add_argument(
        '--pool_strides',
        type=int,
        default=2,
        help='pool_strides')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='embedding_size')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.', )
    parser.add_argument(
        '--desired_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.', )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=10000,
        help='num_steps')
    parser.add_argument(
        '--desired_channels',
        type=int,
        default=1,
        help='desired_channels')
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
        '--learning_rate',
        type=float,
        default=0.01,
        help='learning_rate')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
