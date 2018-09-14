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
