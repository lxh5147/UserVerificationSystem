import os
import unittest

import numpy as np
import tensorflow as tf
from numpy.testing import assert_almost_equal
from python_speech_features import fbank
from scipy.io.wavfile import read

from model.voice_dataset import read_audio, get_input_function, convert_audio_with_PMX, read_audio_int16, \
    _create_feature_generator, _group_by_labels, _rearrange_with_same_label, _random_select, _normalize_frames


class VoiceDatasetTestCase(unittest.TestCase):
    def test_read_audio(self):
        wav_file = '../data/train/121624931534904112937-0.wav'

        audio_val, sample_rate_val, all_samples_val = read_audio(wav_file)
        sample_rate_readed, data_readed = read(wav_file)
        self.assertEqual(all_samples_val, len(data_readed), 'total number of samples')
        self.assertEqual(sample_rate_val, sample_rate_readed, 'sample rate')
        audio_val = audio_val[:all_samples_val]
        self.assertTrue((data_readed == audio_val).all(), 'audio data')

    def test_get_input_function(self):
        wav_files = ['../data/train/121624931534904112937-0.wav',
                     '../data/train/121624931534904112937-1.wav',
                     '../data/train/121624931534904112937-2.wav'
                     ]
        labels = [0, 1, 2]
        desired_ms = 100
        window_size_ms = 25
        window_stride_ms = 10
        batch_size = 2
        features, label_ids = get_input_function(wav_files,
                                                 labels,
                                                 batch_size=batch_size,
                                                 desired_ms=desired_ms,
                                                 window_size_ms=window_size_ms,
                                                 window_stride_ms=window_stride_ms,
                                                 magnitude_squared=True,
                                                 input_feature_dim=40,
                                                 input_feature='fbank',
                                                 is_training=True,
                                                 rearrange_with_same_label=True,
                                                 normalize_frames=False)
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

    def test_create_feature_generator(self):
        wav_files = ['../data/train/121624931534904112937-0.wav']
        labels = [0]
        generator, _ = _create_feature_generator(wav_files, labels,
                                                 window_size_ms=25,
                                                 window_stride_ms=10,
                                                 desired_ms=1000,
                                                 input_feature_dim=40,
                                                 input_feature='fbank',
                                                 is_training=True,
                                                 normalize_frames=False)
        feats_readed = []
        labels_readed = []
        for feat, label in generator():
            feats_readed.append(feat)
            labels_readed.append(label)
        self.assertEqual(labels_readed, labels, 'labels')
        self.assertEqual(len(feats_readed), 1, 'features')
        self.assertEqual(len(feats_readed[0][0]), 40, 'feature dim')

    def test_group_by_labels(self):
        items = [11, 12, 13, 23, 22, 33]
        labels = [1, 1, 1, 2, 2, 3]
        groups = _group_by_labels(items, labels)
        import collections
        groups_expected = collections.OrderedDict({1: [11, 12, 13], 2: [23, 22], 3: [33]})
        self.assertTrue(groups == groups_expected, 'groups')

    def test_rearrange_by_pair(self):
        items = [11, 12, 13, 23, 22, 33]
        labels = [1, 1, 1, 2, 2, 3]
        items_updated, labels_updated = _rearrange_with_same_label(items, labels)
        # <class 'list'>: [3, 2, 2, 1, 1, 1]
        self.assertEqual(sorted(labels), sorted(labels_updated), 'labels')
        self.assertEqual(sorted(items), sorted(items_updated), 'items')

    def test_normalize_frames(self):
        features = np.asanyarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        normed_feature = _normalize_frames(features)
        assert_almost_equal(normed_feature,
                            np.asanyarray([[-1.2247448, 0., 1.2247448], [-1.2247448, 0., 1.2247448]],
                                          dtype=np.float32), decimal=5)

    def test_random_select(self):
        samples = np.asarray([[1], [2], [3]], dtype=np.float32)
        selected = _random_select(samples, 6)
        self.assertTrue((selected == np.asarray([[1], [2], [3], [1], [2], [3]], dtype=np.float32)).all(), 'selected')

    def test_read_audio_and_extract_feature(self):
        wav_file = '../data/train/121624931534904112937-0.wav'
        window_size_ms = 25
        window_stride_ms = 10
        input_feature_dim = 64
        signal, sample_rate, num_samples = read_audio(wav_file)
        feat, _ = fbank(signal,
                        sample_rate,
                        winlen=window_size_ms / 1000,
                        winstep=window_stride_ms / 1000,
                        nfilt=input_feature_dim)
        print(feat)
        feat_normed = _normalize_frames(feat)
        print('==========================================')
        print(feat_normed)


if __name__ == '__main__':
    unittest.main()
