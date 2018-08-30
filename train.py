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
                          desired_channels,
                          magnitude_squared,
                          dct_coefficient_count
                          )

    # Shuffle, repeat, and batch the examples.
    if is_training:
        dataset = dataset.shuffle().repeat().batch(batch_size)

    features, labels = dataset.make_one_shot_iterator().next()
    return features, labels


def create_model(model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None):
    return tf.estimator.Estimator(model_fn,
                                  model_dir=model_dir,
                                  config=config,
                                  params=params,
                                  warm_start_from=warm_start_from
                                  )


def from_ms_to_samples(sample_rate, duration_ms):
    return int(sample_rate * duration_ms / 1000)


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    model = create_model(
        model_dir=FLAGS.model_dir,
        warm_start_from=FLAGS.warm_start_from,
        params={
            'num_channels': FLAGS.num_channels,
            'blocks': FLAGS.blocks,
            'kernel_size': FLAGS.kernel_size,
            'use_batch_norm': FLAGS.use_batch_norm,
            'pool_size': FLAGS.pool_size,
            'pool_strides': FLAGS.pool_strides,
            'embedding_size': FLAGS.embedding_size
        })

    # Define the input function for training
    train_wav_files = get_wav_files(os.path.join(FLAGS.data_dir, 'train'))
    train_labels = get_labels(os.path.join(FLAGS.data_dir, 'train_labels'))
    desired_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.desired_ms)
    window_size_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_size_ms)
    window_stride_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_stride_ms)
    train_input_fn = lambda: input_fn(
        wav_files=train_wav_files,
        labels=train_labels,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=FLAGS.magnitude_squared,
        dct_coefficient_count=FLAGS.dct_coefficient_count
    )

    # model Model
    model.train(train_input_fn, steps=FLAGS.num_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.', )
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='15000,3000',
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--quantize',
        type=bool,
        default=False,
        help='Whether to train the model for eight-bit deployment')
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfcc',
        help='Spectrogram processing mode. Can be "mfcc" or "average"')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
