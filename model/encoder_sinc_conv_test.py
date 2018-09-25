import unittest

import numpy as np
import tensorflow as tf

from model.encoder_cnn import encoder as cnn_encoder
from model.encoder_sinc_conv import encoder, sinc


class SincEncoderTestCase(unittest.TestCase):
    def test_sinc(self):
        band = 2.
        t_right = tf.constant(np.asanyarray([1., 2., 3.], dtype='float32'))
        output = sinc(band, t_right)
        with tf.Session() as sess:
            output_val = sess.run(output)
        self.assertEquals(len(output_val), 7, 'length of output value')
        # TODO: more check

    def test_encoder(self):
        inputs = tf.constant(
            np.asanyarray([[[0.1], [0.2], [0.3], [0.4]], [[0.12], [0.22], [0.32], [0.42]]], dtype='float32'))
        params = {
            'freq_scale': 30,
            'sinc_filters': 20,
            'sinc_kernel_size': 3,
            'filters': [10],
            'blocks': 3,
            'kernel_size': 2,
            'strides': 1,
            'embedding_size': '50'
        }
        output = encoder(inputs, params, cnn_encoder)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output)
        self.assertEquals(output_val.shape, (2, 50), 'output value shape')
        # TODO: more check
        # print(output_val)


if __name__ == '__main__':
    unittest.main()
