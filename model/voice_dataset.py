#  Copyright 2018 AINemo. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the Voice dataset."""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import numpy as np


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.
    Args:
      filename: Path to the .wav file to load.
    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.
    Args:
      filename: Path to save the file to.
      wav_data: 2D array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        sample_rate_placeholder = tf.placeholder(tf.int32, [])
        wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
        wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                               sample_rate_placeholder)
        wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
        sess.run(
            wav_saver,
            feed_dict={
                wav_filename_placeholder: filename,
                sample_rate_placeholder: sample_rate,
                wav_data_placeholder: np.reshape(wav_data, (-1, 1))
            })


def dataset(wav_files, labels,
            window_size_samples,
            window_stride_samples,
            magnitude_squared=True,
            dct_coefficient_count=40):
    '''
    :param wav_files: a list of audio file names
    :param labels: the label of each file
    :return: data set
    '''

    dataset = tf.data.Dataset.from_tensor_slices(
        (wav_files, labels))

    def decode(wav_file):
        wav_loader = io_ops.read_file(wav_file)
        audio, sample_rate = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        spectrogram = contrib_audio.audio_spectrogram(
            audio,
            window_size=window_size_samples,
            stride=window_stride_samples,
            magnitude_squared=magnitude_squared)

        feat = contrib_audio.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count=dct_coefficient_count)

        # TODO: use delta features?

        return feat

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16)
    images = images.map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(directory):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(directory, 'train-images-idx3-ubyte',
                   'train-labels-idx1-ubyte')
