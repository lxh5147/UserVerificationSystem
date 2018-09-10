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
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


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

    def decode(wav_file, _):
        wav_loader = io_ops.read_file(wav_file)
        audio, sample_rate = contrib_audio.decode_wav(wav_loader,
                                                      desired_channels=1)

        # choose a random clip with desired_samples from the audio
        all_samples = tf.shape(audio)[0]
        audio = tf.cond(tf.less(all_samples, desired_samples),
                        true_fn=lambda: tf.pad(tensor=audio,
                                               paddings=[[0, desired_samples - all_samples], [0, 0]],
                                               constant_values=-1),
                        false_fn=lambda: tf.random_crop(value=audio, size=[desired_samples, 1])
                        )
        # update the static shape information of an audio tensor
        audio.set_shape([desired_samples, 1])

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
        return (feat, _)

    return raw_dataset.map(decode)


def from_ms_to_samples(sample_rate, duration_ms):
    return int(sample_rate * duration_ms / 1000)


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
    return label_ids, ids


def get_wav_files(directory):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            files.append(os.path.join(r, file))
    return files


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