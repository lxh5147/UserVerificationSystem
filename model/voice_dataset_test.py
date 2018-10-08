import os
import unittest
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read

from model.voice_dataset import read_audio, get_input_function, convert_audio_with_PMX, read_audio_int16, \
    _create_feature_generator,_post_process_dataset


class VoiceDatasetTestCase(unittest.TestCase):
    def test_read_audio(self):
        wav_file = '../data/train/121624931534904112937-0.wav'
        desired_ms = 10000
        audio_val, sample_rate_val, all_samples_val = read_audio(wav_file,
                                                                 desired_ms)
        sample_rate_readed, data_readed = read(wav_file)
        self.assertEqual(all_samples_val, len(data_readed), 'total number of samples')
        self.assertEqual(sample_rate_val, sample_rate_readed, 'sample rate')
        self.assertEqual(len(audio_val), desired_ms / 1000 * sample_rate_val, 'padded audio length')
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
                                                 is_training=True)
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
                                                 input_feature='fbank')
        feats_readed = []
        labels_readed = []
        for feat, label in generator():
            feats_readed.append(feat)
            labels_readed.append(label)
        self.assertEqual(labels_readed, labels, 'labels')
        self.assertEqual(len(feats_readed), 1, 'features')
        self.assertEqual(len(feats_readed[0][0]), 40, 'feature dim')
    def test_dataset_in_out(self):
        wav_files = ['../data/train/121624931534904112937-0.wav']
        labels = [0]
        generator, output_shapes = _create_feature_generator(wav_files, labels,
                                                 window_size_ms=25,
                                                 window_stride_ms=10,
                                                 desired_ms=1000,
                                                 input_feature_dim=40,
                                                 input_feature='fbank')
        feats_readed = []
        labels_readed = []
        for feat_before, label_before in generator():
            label_before=np.array(label_before)
            feats_readed.append(feat_before)
            labels_readed.append(label_before)
        voice_dataset = tf.data.Dataset.from_generator(
            generator,
            (tf.float32, tf.int64),
            output_shapes=output_shapes
        )
        voice_dataset = _post_process_dataset(voice_dataset,
                                              batch_size=1,
                                              is_training=True)
        features_after, labels_after = voice_dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            features_after, labels_after = sess.run([features_after, labels_after])
        feats_readed.append(features_after)
        labels_readed.append(labels_after)
        print(feats_readed[0]==feats_readed[1])
        print(labels_readed[0]==labels_readed[1])


if __name__ == '__main__':
    unittest.main()
