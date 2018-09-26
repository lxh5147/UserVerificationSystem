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
import collections
import os
from random import shuffle

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

    raw_dataset = tf.data.Dataset.from_generator(_create_generator(wav_files, labels),
                                                 (tf.string, tf.int64),
                                                 (tf.TensorShape([]), tf.TensorShape([])))

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


def dataset_raw(wav_files,
                labels,
                desired_samples):
    '''
    :param wav_files: a list of audio file names
    :param labels: the label of each file
    :param desired_samples: how many number of samples to load from a wav file.
    :return: raw data set
    '''
    raw_dataset = tf.data.Dataset.from_generator(_create_generator(wav_files, labels),
                                                 (tf.string, tf.int64),
                                                 (tf.TensorShape([]), tf.TensorShape([])))

    def decode(wav_file, label):
        audio, sample_rate, _ = read_audio(wav_file, desired_samples)
        return (audio, label)

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


def _post_process_dataset(dataset,
                          batch_size,
                          is_training=True):
    if is_training:
        dataset = dataset.repeat().batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    return dataset


def _input_fn_raw(wav_files,
                  labels,
                  is_training=True,
                  **kwargs):
    batch_size = kwargs['batch_size']
    desired_samples = kwargs['desired_samples']
    voice_dataset = dataset_raw(wav_files,
                                labels,
                                desired_samples
                                )
    voice_dataset = _post_process_dataset(
        voice_dataset,
        batch_size,
        is_training)
    audios, labels = voice_dataset.make_one_shot_iterator().get_next()
    return audios, labels


def _input_fn_feature(wav_files,
                      labels,
                      is_training=True,
                      **kwargs):
    batch_size = kwargs['batch_size']
    desired_samples = kwargs['desired_samples']
    window_size_samples = kwargs['window_size_samples']
    window_stride_samples = kwargs['window_stride_samples']
    magnitude_squared = kwargs['magnitude_squared']
    dct_coefficient_count = kwargs['dct_coefficient_count']
    voice_dataset = dataset(wav_files,
                            labels,
                            desired_samples,
                            window_size_samples,
                            window_stride_samples,
                            magnitude_squared,
                            dct_coefficient_count
                            )
    voice_dataset = _post_process_dataset(voice_dataset,
                                          batch_size,
                                          is_training)
    features, labels = voice_dataset.make_one_shot_iterator().get_next()
    return features, labels


def get_input_function(
        wav_files,
        labels,
        is_training=True,
        encoder='cnn',
        **kwargs
):
    # return the input function for a given type of encoder
    assert encoder in ['cnn', 'resnet', 'sinc_cnn', 'sinc_resnet']
    if encoder in ['cnn', 'resnet']:
        return lambda: _input_fn_feature(wav_files,
                                     labels,
                                     is_training,
                                     **kwargs)
    elif encoder in ['sinc_cnn', 'sinc_resnet']:
        return lambda: _input_fn_raw(wav_files,
                                         labels,
                                         is_training,
                                         **kwargs
                                         )


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


def _group_by_labels(items, labels):
    '''
    Group items by labels. Both the label and item are in the same order as they occur.
    :param items: a list of Objects to be grouped
    :param labels: the corresponding item labels
    :return: an ordered dictionary, representing the grouped items
    '''
    groups = collections.OrderedDict()
    for item, label in zip(items, labels):
        if label in groups:
            groups[label].append(item)
        else:
            groups[label] = [item]
    return groups


def _rearrange_with_same_label(items, labels, n=2):
    '''
    Re-arrange items so that n items have the same label and the next n with different label.
    :param items: a list of items to arrange
    :param labels: item labels
    :param n: int, the expected number of items with the same label
    :return: a list of items re-arranged
    '''
    groups = _group_by_labels(items, labels)
    items_updated = []
    labels_updated = []
    labels_ordered = groups.keys()
    counts_readed = dict()
    for label in labels_ordered:
        counts_readed[label] = 0
    while len(items_updated) < len(items):
        # rearrange all items
        for label in labels_ordered:
            group = groups[label]
            count = len(group)
            count_readed = counts_readed[label]
            # try to read a pair of items from a group
            for i in range(n):
                if count_readed < count:
                    items_updated.append(group[count_readed])
                    labels_updated.append(label)
                    count_readed += 1
            counts_readed[label] = count_readed
    return items_updated, labels_updated


def _shuffle_and_rearrange_with_same_label(items, labels, n=2):
    # re-shuffle the items and try to put every two items with the same label
    zipped = list(zip(items, labels))
    shuffle(zipped)
    _items, _labels = tuple(zip(*zipped))
    return _rearrange_with_same_label(_items, _labels, n)


def _create_generator(items, labels):
    # create a generator for the dataset
    def generator():
        _items, _labels = _shuffle_and_rearrange_with_same_label(items, labels)
        for item, label in zip(_items, _labels):
            yield (item, label)

    return generator
