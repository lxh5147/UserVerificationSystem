import os
import unittest

import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read

from model.voice_dataset import read_audio, input_fn, convert_audio_with_PMX, read_audio_int16


class VoiceDatasetTestCase(unittest.TestCase):
    def test_read_audio(self):
        wav_file_val = '../data/train/121624931534904112937-0.wav'
        desired_samples = 100000
        wav_file = tf.placeholder(dtype=tf.string)
        audio, sample_rate, all_samples = read_audio(wav_file,
                                                     desired_samples)
        with tf.Session() as sess:
            all_samples_val, audio_val, sample_rate_val = sess.run([all_samples, audio, sample_rate],
                                                                   feed_dict={wav_file: wav_file_val})

        sample_rate_readed, data_readed = read(wav_file_val)
        self.assertEqual(all_samples_val, len(data_readed), 'total number of samples')
        self.assertEqual(sample_rate_val, sample_rate_readed, 'sample rate')
        self.assertEqual(len(audio_val), desired_samples, 'padded audio length')
        audio_val = audio_val[:all_samples_val, 0]
        audio_val = (32767 * audio_val).astype('int16')
        self.assertTrue(np.max(np.abs(data_readed - audio_val)) <= 1, 'audio data')

    def test_input_fn(self):
        wav_files = ['../data/train/121624931534904112937-0.wav',
                     '../data/train/121624931534904112937-1.wav',
                     '../data/train/121624931534904112937-2.wav'
                     ]
        labels = [0, 1, 2]
        desired_samples = 1600
        window_size_samples = 400
        window_stride_samples = 100
        batch_size = 2
        features, label_ids = input_fn(wav_files,
                                       labels,
                                       batch_size=batch_size,
                                       desired_samples=desired_samples,
                                       window_size_samples=window_size_samples,
                                       window_stride_samples=window_stride_samples,
                                       magnitude_squared=True,
                                       dct_coefficient_count=40,
                                       is_training=True,
                                       buffer_size=1000)
        labels_readout = []
        repeated_times = 10
        with tf.Session() as sess:
            for i in range(repeated_times):
                label_ids_val = sess.run(label_ids)
                labels_readout.extend(label_ids_val)

        self.assertEqual(len(labels_readout), repeated_times * batch_size, 'total number of labels')
        self.assertEqual(list(dict.fromkeys(labels_readout)).sort(), list(dict.fromkeys(labels)).sort(),
                         'the same unique labels')

    def test_convert_audio_with_PMX(self):
        wav_file = '../data/test/1252695_voice_reco_1527810946756.wav'
        wav_file_processed = '../data/test/1252695_voice_reco_1527810946756_processed.wav'
        convert_audio_with_PMX(wav_file, wav_file_processed)
        # this customized function ignores the PMX chunk
        data, sr = read_audio_int16(wav_file)
        sr_readed, data_readed = read(wav_file_processed)
        self.assertEqual(sr, sr_readed, 'sample rate')
        self.assertTrue((data == data_readed).all(), 'audio data')
        os.remove(wav_file_processed)


if __name__ == '__main__':
    unittest.main()
