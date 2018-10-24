import os
import unittest

import numpy as np
import tensorflow as tf

import service
from model.model_fn import create_model
from model.voice_dataset import from_ms_to_samples
from service import _write_pcm16_wav, _parse_environ, _get_device_root_path, _get_user_root_path

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
            'embedding_size': 512,
            'encoder': 'rescnn',
            'sample_rate': 16000,
            'desired_samples': from_ms_to_samples(16000, 1000),
            'window_size_samples': from_ms_to_samples(16000, 30.0),
            'window_stride_samples': from_ms_to_samples(16000, 10.0),
            'magnitude_squared': True,
            'dct_coefficient_count': 64,
            'batch_size': 10,
            'threshold': 0.2,
            'normalize_frames':False
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
        os.remove(output_file)

    def test_parse_environ(self):
        import base64
        import json
        data = 'some data'

        request_body = {
            'device_id': 'family1',
            'user_id': 'user1',
            'func_id': service.FUNC_ENROLL,
            'streams': [base64.b64encode(data.encode()).decode()]
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
        self.assertEqual(len(streams), 1, 'streams')
        data_readed = streams[0].decode()
        self.assertEqual(data, data_readed, 'stream')

    def test_get_device_root_path(self):
        device_id = 'family1'
        self.assertEqual(_get_device_root_path(device_id), \
                         os.path.join(service.FLAGS.data_dir, '__device_' + device_id), \
                         'device root path')

    def test_get_user_root_path(self):
        device_id = 'family1'
        user_id = 'user1'
        self.assertEqual(_get_user_root_path(device_id, user_id), \
                         os.path.join(service.FLAGS.data_dir, '__device_' + device_id, '__user_' + user_id), \
                         'user root path')


if __name__ == '__main__':
    unittest.main()
