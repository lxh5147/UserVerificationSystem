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
from math import ceil
from random import shuffle, randint

import audioread
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from python_speech_features import fbank, logfbank, mfcc
from scipy.io.wavfile import write


def read_audio(wav_file):
    sample_rate, signal = wav.read(wav_file)
    # one channel wav
    assert np.ndim(signal) == 1
    num_samples = len(signal)
    return signal, sample_rate, num_samples


def _random_select(samples, num_to_select):
    num_samples = len(samples)
    if num_samples < num_to_select:
        pad_width = [(0, num_to_select - num_samples)]
        for _ in range(samples.ndim - 1):
            pad_width.append((0, 0))
        samples_selected = np.pad(samples,
                                  pad_width,
                                  mode='wrap'
                                  )
    else:
        offset = randint(0, num_samples - num_to_select)
        samples_selected = samples[offset:offset + num_to_select]
    return samples_selected


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


'''
TODO: optimize the performance: https://www.tensorflow.org/performance/datasets_performance
Here is a summary of the best practices for designing input pipelines:

    Use the prefetch transformation to overlap the work of a producer and consumer. In particular, we recommend adding prefetch(n) (where n is the number of elements / batches consumed by a training step) to the end of your input pipeline to overlap the transformations performed on the CPU with the training done on the accelerator.
    Parallelize the map transformation by setting the num_parallel_calls argument. We recommend using the number of available CPU cores for its value.
    If you are combining pre-processed elements into a batch using the batch transformation, we recommend using the fused map_and_batch transformation; especially if you are using large batch sizes.
    If you are working with data stored remotely and / or requiring deserialization, we recommend using the parallel_interleave transformation to overlap the reading (and deserialization) of data from different files.
    Vectorize cheap user-defined functions passed in to the map transformation to amortize the overhead associated with scheduling and executing the function.
    If your data can fit into memory, use the cache transformation to cache it in memory during the first epoch, so that subsequent epochs can avoid the overhead associated with reading, parsing, and transforming it.
    If your pre-processing increases the size of your data, we recommend applying the interleave, prefetch, and shuffle first (if possible) to reduce memory usage.
    We recommend applying the shuffle transformation before the repeat transformation, ideally using the fused shuffle_and_repeat transformation.

'''


def _post_process_dataset(dataset,
                          batch_size,
                          is_training=True):
    if is_training:
        # we found using map_and_batch and dataset.prefetch in the input function gives a solid boost in performance.
        # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size.
        dataset = dataset.repeat().batch(batch_size).prefetch(buffer_size=None)
    else:
        dataset = dataset.batch(batch_size)
    return dataset


def get_input_function(
        wav_files,
        labels,
        is_training=True,
        **kwargs
):
    # return the input function for a given type of encoder
    batch_size = kwargs['batch_size']
    generator, output_shapes = _create_feature_generator(wav_files,
                                                         labels,
                                                         is_training,
                                                         **kwargs)
    voice_dataset = tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.int64),
        output_shapes=output_shapes
    )
    voice_dataset = _post_process_dataset(voice_dataset,
                                          batch_size,
                                          is_training)
    features, labels = voice_dataset.make_one_shot_iterator().get_next()
    return features, labels


def read_audio_int16(path):
    # read 16-bits signed int wav data
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
    :param items: a list of objects to be grouped
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
    labels_ordered = list(groups.keys())
    counts_readed = dict()
    for label in labels_ordered:
        counts_readed[label] = 0
    while len(items_updated) < len(items):
        # choose a label
        label_index = randint(0, len(labels_ordered) - 1)
        label = labels_ordered[label_index]
        group = groups[label]
        count = len(group)
        count_readed = counts_readed[label]
        # try to read n items for this label
        if count_readed < count:
            for i in range(n):
                if count_readed < count:
                    items_updated.append(group[count_readed])
                    labels_updated.append(label)
                    count_readed += 1
            counts_readed[label] = count_readed
    return items_updated, labels_updated


def _shuffle(items, labels):
    # re-shuffle the items and labels
    zipped = list(zip(items, labels))
    shuffle(zipped)
    _items, _labels = tuple(zip(*zipped))
    _items, _labels = _rearrange_with_same_label(_items, _labels)
    return _items, _labels


def _normalize_frames(features, epsilon=1e-12):
    '''
       Norm the features per frame across all features.
       :param features: ndarray, shape (NUM_FRAMES, dim)
       :param epsilon: float to avoid div by zero
       :return: normed frame features
       '''
    mean_scale = np.mean(features, axis=1)
    std_scale = np.maximum(np.std(features, axis=1), epsilon)
    features = (features - mean_scale[:, np.newaxis]) / std_scale[:, np.newaxis]
    return features


def _create_feature_generator(wav_files, labels, is_training=True, **kwargs):
    window_size_ms = kwargs['window_size_ms']
    window_stride_ms = kwargs['window_stride_ms']
    desired_ms = kwargs['desired_ms']
    input_feature_dim = kwargs['input_feature_dim']
    input_feature = kwargs['input_feature']
    normalize_frames = kwargs.get('normalize_frames', True)
    assert input_feature in ['mfcc', 'fbank', 'logfbank', 'raw']

    def generator():
        if is_training:
            _wav_files, _labels = _shuffle(wav_files, labels)
        else:
            _wav_files, _labels = wav_files, labels

        for wav_file, label in zip(_wav_files, _labels):
            signal, sample_rate, _ = read_audio(wav_file)
            num_frames = ceil(desired_ms / window_stride_ms)
            num_samples = from_ms_to_samples(desired_ms, sample_rate)
            if input_feature == 'fbank':
                feat, _ = fbank(signal,
                                sample_rate,
                                winlen=window_size_ms / 1000,
                                winstep=window_stride_ms / 1000,
                                nfilt=input_feature_dim)
                feat = _random_select(feat, num_frames)
            elif input_feature == 'logfbank':
                feat = logfbank(signal,
                                sample_rate,
                                winlen=window_size_ms / 1000,
                                winstep=window_stride_ms / 1000,
                                nfilt=input_feature_dim)
                feat = _random_select(feat, num_frames)
            elif input_feature == 'mfcc':
                feat = mfcc(signal,
                            sample_rate,
                            winlen=window_size_ms / 1000,
                            winstep=window_stride_ms / 1000,
                            nfilt=input_feature_dim,
                            numcep=input_feature_dim)
                feat = _random_select(feat, num_frames)
            elif input_feature == 'raw':
                feat = np.expand_dims(signal, 1)
                feat = _random_select(feat, num_samples)

            # norm per dimension across all frames
            if normalize_frames:
                feat = _normalize_frames(feat)
            yield (feat, label)

    if input_feature in ['fbank', 'logfbank', 'mfcc']:
        output_shapes = (tf.TensorShape([None, input_feature_dim]), tf.TensorShape([]))
    elif input_feature in ['raw']:
        output_shapes = (tf.TensorShape([None, 1]), tf.TensorShape([]))
    return generator, output_shapes
