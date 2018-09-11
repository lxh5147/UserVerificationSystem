import unittest
import tensorflow as tf
from model.voice_dataset import read_audio

class VoiceDatasetTestCase(unittest.TestCase):
    def test_read_audio(self):

        wav_file_val='../data/train/121624931534904112937-0.wav'
        desired_samples = 100000
        wav_file = tf.placeholder(dtype=tf.string)
        audio, sample_rate, all_samples = read_audio(wav_file,
                                                     desired_samples)
        with tf.Session() as sess:
            all_samples_val, audio_val = sess.run([all_samples, audio],
                                                  feed_dict={wav_file: wav_file_val})

        self.assertTrue(all_samples_val < desired_samples,'less than desired samples')
        self.assertEqual(len(audio_val),desired_samples,'padded audio length')


if __name__ == '__main__':
    unittest.main()
