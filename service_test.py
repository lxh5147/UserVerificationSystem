import os
import unittest

import numpy as np
import tensorflow as tf

import service
from model.model_fn import create_model
from service import _write_pcm16_wav, _parse_environ

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

    def test_parse_environ(self):
        import base64
        import json
        request_body = {
            'device_id': 'family1',
            'user_id': 'user1',
            'func_id': service.FUNC_ENROLL
        }
        content = json.dumps(request_body)
        import io

        input = io.BytesIO()
        input.write(content.encode())
        input.seek(0)
        environ = {
            'wsgi.input': input,
            'CONTENT_LENGTH': len(content)
        }
        device_id, user_id, func_id, streams = _parse_environ(environ)
        self.assertEqual(device_id, request_body['device_id'], 'device_id')
        self.assertEqual(user_id, request_body['user_id'], 'user_id')
        self.assertEqual(func_id, request_body['func_id'], 'func_id')
        self.assertEqual(len(streams), 0, 'streams')


if __name__ == '__main__':
    unittest.main()
