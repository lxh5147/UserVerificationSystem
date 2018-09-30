import argparse
import os
import sys

import tensorflow as tf

from model.model_fn import create_model
from model.voice_dataset import get_file_and_labels, get_input_function


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)
    # Define the input function for training
    wav_files, labels, label_to_id = get_file_and_labels(os.path.join(FLAGS.data_dir, 'train_labels'))
    wav_files = [os.path.join(FLAGS.data_dir, 'train', wav_file) for wav_file in wav_files]

    train_num_classes = len(label_to_id)

    model = create_model(
        model_dir=FLAGS.model_dir,
        params={
            'num_classes': train_num_classes,
            **FLAGS.__dict__
        })
    train_input_fn = lambda: get_input_function(
        wav_files=wav_files,
        labels=labels,
        **FLAGS.__dict__
    )
    model.train(train_input_fn,
                steps=FLAGS.num_steps)


if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=1 python train.py --model_dir='../puffer_515' --data_dir='../../UserVerificationSystem/data/TRAIN/puffer515' --encoder='resnet' --input_feature='fbank' --input_feature_dim=40 --filters=64 128 256 512 --blocks=3 --kernel_size=3 --strides=2 --embedding_size=512 --window_size_ms=25 --desired_ms=1200 --window_stride_ms=10 --magnitude_squared=True --feature=40 --batch_size=160 --triplet_strategy='batch_hard' --margin=0.2 --squared=True --num_steps=126000 --learning_rate=0.01 --learning_rate_decay_rate=0.5 --learning_rate_decay_steps=2520 --l2_regularization_weight=0.00001 --triplet_loss_weight=1 --cross_entropy_loss_weight=0
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
        '--input_feature',
        type=str,
        default='fbank',
        help='Input feature: Use raw|mfcc|fbank|logfbank. Only raw is valid if the encoder is sinc_*')
    parser.add_argument(
        '--encoder',
        type=str,
        default='cnn',
        help='Encoder that encodes a wav to a vector. Use cnn|resnet|sinc_cnn|sinc_resnet')
    parser.add_argument(
        '--sinc_freq_scale',
        type=float,
        default=16000.,
        help='Frequency scale of sinc input feature extractor')
    parser.add_argument(
        '--sinc_kernel_size',
        type=int,
        default=3,
        help='Kernel size of sinc input feature extractor')
    parser.add_argument(
        '--filters',
        type=int,
        nargs='+',
        default=[64, 128, 256, 512],
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
        '--input_feature_dim',
        type=int,
        default=40,
        help='Dimension of input feature')
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
