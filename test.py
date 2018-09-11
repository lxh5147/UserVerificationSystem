import tensorflow as tf
import numpy as np
from model.voice_dataset import read_audio

import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


def read_audio(wav_file, desired_samples, desired_channels=1):
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
                                                  desired_samples=desired_samples,
                                                  desired_channels=desired_channels)

    return audio, sample_rate, sample_rate

    # choose a random clip with desired_samples from the audio
    num_samples = tf.shape(audio)[0]
    audio = tf.cond(tf.less(num_samples, desired_samples),
                    true_fn=lambda: tf.pad(tensor=audio,
                                           paddings=[[0, desired_samples - num_samples], [0, 0]],
                                           constant_values=-1),
                    false_fn=lambda: tf.random_crop(value=audio, size=[desired_samples, 1])
                    )
    # update the static shape information of an audio tensor
    audio.set_shape([desired_samples, 1])
    return audio, sample_rate, num_samples

from scipy.io.wavfile import read,write
import librosa
#1252695_voice_reco_1527810946756.wav
def test_wav_read():
    wav_file_val = './data/test/1252695_voice_reco_1527810946756.wav'
    wav_file_val_processed = './data/test/1252695_voice_reco_1527810946756_processed.wav'

    data,sample_rate = librosa.load(wav_file_val,sr=16000)
    # convert the data to int16
    data = data.astype(dtype='int16')

    write(wav_file_val_processed,sample_rate,data)

    sample_rate, data = read(wav_file_val_processed)
    print(sample_rate, len(data))
    desired_samples = 1000
    wav_file = tf.placeholder(dtype=tf.string)
    audio, sample_rate, all_samples = read_audio(wav_file,
                                                 desired_samples)
    with tf.Session() as sess:
        all_samples_val, audio_val = sess.run([all_samples, audio],
                                              feed_dict={wav_file: wav_file_val_processed})
        print(all_samples_val)
test_wav_read()