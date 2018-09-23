import argparse
import os
import sys
from random import shuffle

import tensorflow as tf

from model.model_fn import create_model
from model.voice_dataset import input_fn, get_file_and_labels, from_ms_to_samples


def _shuffle_and_rearrange_by_pair(items, labels):
    zipped = list(zip(items, labels))
    shuffle(zipped)
    _items, _labels = tuple(zip(*zipped))
    return list(_items),list(_labels)

def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the input function for training
    wav_files, labels, label_to_id = get_file_and_labels(os.path.join(FLAGS.data_dir, 'train_labels'))
    wav_files, labels = _shuffle_and_rearrange_by_pair(wav_files, labels)
    wav_files = [os.path.join(FLAGS.data_dir, 'train', wav_file) for wav_file in wav_files]

    train_num_classes = len(label_to_id)
    filters = map(lambda _: int(_), FLAGS.filters.split(','))
    model = create_model(
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

    desired_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.desired_ms)
    window_size_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_size_ms)
    window_stride_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_stride_ms)
    train_input_fn = lambda: input_fn(
        wav_files=wav_files,
        labels=labels,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=FLAGS.magnitude_squared,
        dct_coefficient_count=FLAGS.dct_coefficient_count,
        batch_size=FLAGS.batch_size
    )
    model.train(train_input_fn,
                steps=FLAGS.num_steps)


if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=1 python train.py --model_dir='../puffer_515' --data_dir='../../UserVer/UserVerificationSystem/data' --encoder='resnet'  --filters='64,128,256,512' --blocks=3 --kernel_size=3 --strides=2 --embedding_size=512 --sample_rate=16000 --window_size_ms=25 --desired_ms=1200 --window_stride_ms=10 --magnitude_squared=True --dct_coefficient_count=40 --batch_size=30 --triplet_strategy='batch_hard' --margin=0.2 --squared=True --num_steps=12600 --learning_rate=0.01 --learning_rate_decay_rate=0.5 --learning_rate_decay_steps=2520 --l2_regularization_weight=0.00001 --triplet_loss_weight=1 --cross_entropy_loss_weight=0
    tensorboard --logdir=../puffer_515/
    '''
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
        help='data dir')
    parser.add_argument(
        '--encoder',
        type=str,
        default='cnn',
        help='Encoder that encodes a wav to a vector. Use cnn or resnet')
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

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + _)
