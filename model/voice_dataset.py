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
import os


def _preprocess_wav(wav_input_filename_placeholder,
                    wav_output_filename_placeholder):
    wav_loader = io_ops.read_file(wav_input_filename_placeholder)
    audio, sample_rate = contrib_audio.decode_wav(wav_loader)
    # TODO: further preprocess; even consider feature extraction
    wav_encoder = contrib_audio.encode_wav(audio, sample_rate)
    wav_saver = io_ops.write_file(wav_output_filename_placeholder, wav_encoder)
    return wav_saver


def _preprocess_wav_files(directory,
                          output_directory):
    with tf.Session(graph=tf.Graph()) as sess:

        wav_input_filename_placeholder = tf.placeholder(tf.string, [])
        wav_output_filename_placeholder = tf.placeholder(tf.string, [])
        wav_saver = _preprocess_wav(wav_input_filename_placeholder,
                                    wav_output_filename_placeholder)

        for folder, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.wav'):
                    wav_input_filename = os.path.join(folder, filename)
                    wav_output_filename = os.path.join(output_directory, filename)
                    sess.run(
                        wav_saver,
                        feed_dict={
                            wav_input_filename_placeholder: wav_input_filename,
                            wav_output_filename_placeholder: wav_output_filename
                        })


def dataset(wav_files,
            labels,
            desired_samples,
            window_size_samples,
            window_stride_samples,
            desired_channels=1,
            magnitude_squared=True,
            dct_coefficient_count=40):
    '''
    :param wav_files: a list of audio file names
    :param labels: the label of each file
    :param desired_samples: how many number of samples to load from a wav file.
    :param window_size_samples: how wide the input window is in samples
    :param window_stride_samples:how widely apart the center of adjacent sample windows should be
    :param magnitude_squared: Whether to return the squared magnitude or just the
      magnitude. Using squared magnitude can avoid extra calculations.
    :param dct_coefficient_count: How many output channels to produce per time slice.
    :return: data set
    '''

    raw_dataset = tf.data.Dataset.from_tensor_slices(
        (wav_files, labels))

    def decode(wav_file, _):
        wav_loader = io_ops.read_file(wav_file)
        audio, sample_rate = contrib_audio.decode_wav(wav_loader,
                                                      desired_samples=desired_samples,
                                                      desired_channels=desired_channels)
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

        return (feat, _)

    return raw_dataset.map(decode)
