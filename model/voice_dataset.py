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
from random import shuffle, randint

import audioread
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from python_speech_features import fbank, logfbank, mfcc
from scipy.io.wavfile import write


def read_audio(wav_file, desired_ms):
    sample_rate, signal = wav.read(wav_file)
    # one channel wav
    assert np.ndim(signal) == 1

    num_samples = len(signal)
    if desired_ms > 0:
        desired_samples = from_ms_to_samples(sample_rate, desired_ms)
        if num_samples < desired_samples:
            signal = np.pad(signal,
                            (0, desired_samples - num_samples),
                            mode='minimum'
                            )
        else:
            offset = randint(0, num_samples - desired_samples)
            signal = signal[offset:offset + desired_samples]
    return signal, sample_rate, num_samples


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
    generator, output_shapes = _create_feature_generator(wav_files, labels,
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


def _shuffle(items, labels):
    # re-shuffle the items and labels
    zipped = list(zip(items, labels))
    shuffle(zipped)
    _items, _labels = tuple(zip(*zipped))
    return _items, _labels


def _create_feature_generator(wav_files, labels, **kwargs):
    window_size_ms = kwargs['window_size_ms']
    window_stride_ms = kwargs['window_stride_ms']
    desired_ms = kwargs['desired_ms']
    input_feature_dim = kwargs['input_feature_dim']
    input_feature = kwargs['input_feature']
    assert input_feature in ['mfcc', 'fbank', 'logfbank', 'raw']

    def generator():
        _wav_files, _labels = _shuffle(wav_files, labels)
        for wav_file, label in zip(_wav_files, _labels):
            signal, sample_rate, _ = read_audio(wav_file, desired_ms)
            if input_feature == 'fbank':
                feat, _ = fbank(signal,
                                sample_rate,
                                winlen=window_size_ms / 1000,
                                winstep=window_stride_ms / 1000,
                                nfilt=input_feature_dim)
            elif input_feature == 'logfbank':
                feat = logfbank(signal,
                                sample_rate,
                                winlen=window_size_ms / 1000,
                                winstep=window_stride_ms / 1000,
                                nfilt=input_feature_dim)
            elif input_feature == 'mfcc':
                feat = mfcc(signal,
                            sample_rate,
                            winlen=window_size_ms / 1000,
                            winstep=window_stride_ms / 1000,
                            nfilt=input_feature_dim,
                            numcep=input_feature_dim)
            elif input_feature == 'raw':
                feat = np.expand_dims(signal, 1)
            yield (feat, label)

    if input_feature in ['fbank', 'logfbank', 'mfcc']:
        output_shapes = (tf.TensorShape([None, input_feature_dim]), tf.TensorShape([]))
    elif input_feature in ['raw']:
        output_shapes = (tf.TensorShape([None, 1]), tf.TensorShape([]))
    return generator, output_shapes
