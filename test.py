import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

def test_audio_too_short():
    desired_samples=100000
    wav_file = "./puffer_data/train/1252695_voice_reco_1527810402837.wav"
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
    with tf.Session() as sess:
        
        all_samples_val, audio_val = sess.run([all_samples, audio], feed_dict={
            wav_file: "./puffer_data/train/1252695_voice_reco_1527810402837.wav"})
        print("length:{} padded length:{}".format(all_samples_val, len(audio_val)))

test_audio_too_short()