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
import os

import audioread
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


def read_audio(wav_file, desired_samples=-1, desired_channels=1):
    '''
    :param wav_file: a string tensor that represents the target wav file
    :param desired_samples: how many samples to read from the wav file
    :param desired_channels: the number of channels to read from the wav file
    :return: a tuple of audio, sample_rate, num_samples,
             where audio is a tensor of type float32 with shape (desired_samples, desired_channels),
             sample_rate is a scale tensor of int32 that represents the sample rate of this audio,
             and num_samples is a scale tensor of int32 that represents the total number of samples of this audio
    '''

    wav_loader = io_ops.read_file(wav_file)
    audio, sample_rate = contrib_audio.decode_wav(wav_loader,
                                                  desired_channels=desired_channels)
    num_samples = tf.shape(audio)[0]

    if desired_samples >= 0:  # otherwise we read all the samples
        # choose a random clip with desired_samples from the audio
        audio = tf.cond(tf.less(num_samples, desired_samples),
                        true_fn=lambda: tf.pad(tensor=audio,
                                               paddings=[[0, desired_samples - num_samples], [0, 0]],
                                               constant_values=-1),
                        false_fn=lambda: tf.random_crop(value=audio, size=[desired_samples, 1])
                        )
        # update the static shape information of an audio tensor
        audio.set_shape([desired_samples, 1])

    return audio, sample_rate, num_samples


def extract_audio_feature(audio,
                          sample_rate,
                          window_size_samples,
                          window_stride_samples,
                          magnitude_squared=True,
                          dct_coefficient_count=40):
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
    # from 1, time_steps, dct_coefficient_count to time steps, dct_coefficient_count
    feat = tf.squeeze(feat)
    return feat


def dataset(wav_files,
            labels,
            desired_samples,
            window_size_samples,
            window_stride_samples,
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

    def decode(wav_file, label):
        audio, sample_rate, _ = read_audio(wav_file, desired_samples)
        feat = extract_audio_feature(audio,
                                     sample_rate,
                                     window_size_samples=window_size_samples,
                                     window_stride_samples=window_stride_samples,
                                     magnitude_squared=magnitude_squared,
                                     dct_coefficient_count=dct_coefficient_count)
        return (feat, label)

    return raw_dataset.map(decode)


def from_ms_to_samples(sample_rate, duration_ms):
    return int(sample_rate * duration_ms / 1000)


def get_file_and_labels(file_and_labels_file):
    with open(file_and_labels_file) as f:
        lines = f.read().splitlines()

    label_to_id = {}
    label_ids = []
    files = []
    for line in lines:
        file, label = line.split(',')
        files.append(file)
        if label in label_to_id:
            cur_id = label_to_id[label]
            label_ids.append(cur_id)
        else:
            cur_id = len(label_to_id)
            label_to_id[label] = cur_id
            label_ids.append(cur_id)

    return files, label_ids, label_to_id


def input_fn(wav_files,
             labels,
             batch_size,
             desired_samples,
             window_size_samples,
             window_stride_samples,
             magnitude_squared=True,
             dct_coefficient_count=40,
             is_training=True,
             buffer_size=1000):
    voice_dataset = dataset(wav_files,
                            labels,
                            desired_samples,
                            window_size_samples,
                            window_stride_samples,
                            magnitude_squared,
                            dct_coefficient_count
                            )

    # Shuffle, repeat, and batch the examples.
    if is_training:
        voice_dataset = voice_dataset.shuffle(buffer_size=buffer_size).repeat().batch(batch_size)
    else:
        voice_dataset = voice_dataset.batch(batch_size)

    features, labels = voice_dataset.make_one_shot_iterator().get_next()
    return features, labels


def read_audio_int16(path):
    fmt = '<i{:d}'.format(2)
    dtype = 'int16'
    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        for frame in input_file:
            frame = np.frombuffer(frame, fmt).astype(dtype)
            y.append(frame)
        if y:
            y = np.concatenate(y)
        # Final cleanup for dtype and contiguity
        y = np.ascontiguousarray(y, dtype=dtype)
        return (y, sr_native)


def convert_audio_with_PMX(input_wav, output_wav):
    '''
    remove the chunks added by Adobe
    :param input_wav: input wav file that contains _PMX chunks
    :param output_wav: a wav with the _PMX chunks removed
    :return: None
    '''
    data, sample_rate = read_audio_int16(input_wav)
    write(output_wav, sample_rate, data)
