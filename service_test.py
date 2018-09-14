import unittest
import tensorflow as tf
from model.model_fn import create_model
import service
from service import _write_pcm16_wav
import os
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def _to_object_with_attributes(d):
    t = lambda: None
    for k in d:
        setattr(t, k, d[k])
    return t


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        service.FLAGS = _to_object_with_attributes({
            'model_dir': './tmp_model',
            'data_dir': './data/online',
            'filters': [64, 128, 256, 512],
            'blocks': 3,
            'kernel_size': 3,
            'strides': 2,
            'embedding_size': 128,
            'encoder': 'cnn',
            'desired_ms': 1000,
            'window_size_ms': 30.0,
            'window_stride_ms': 10.0,
            'sample_rate': 16000,
            'magnitude_squared': True,
            'dct_coefficient_count': 40,
            'batch_size': 10,
            'threshold': 0.2
        })

        service.model = create_model(
            model_dir=service.FLAGS.model_dir,
            params={
                'filters': service.FLAGS.filters,
                'blocks': service.FLAGS.blocks,
                'kernel_size': service.FLAGS.kernel_size,
                'strides': service.FLAGS.strides,
                'embedding_size': service.FLAGS.embedding_size,
                'encoder': service.FLAGS.encoder
            })
        service.grouped_registerations = dict()

    def test_write_pcm16_wav(self):
        data = np.asanyarray([12, 13, 15, 18], dtype='int16')
        output_file = './data/test/fake.wav'
        _write_pcm16_wav(output_file, data)
        self.assertTrue(os.path.exists(output_file))
        # read the audio
        import scipy.io.wavfile as wavefile
        _, data_readed = wavefile.read(output_file)
        self.assertTrue((data == data_readed).all(), 'audio data equal')


if __name__ == '__main__':
    unittest.main()
