import unittest

import numpy as np
import tensorflow as tf

from model.encoder_cnn import encoder


class AttentionTestCase(unittest.TestCase):
    def test_encoder_cnn(self):
        batch_size=2
        inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size,2,3))
        params={}
        params['filters']=[64]
        params['blocks']=1
        params['kernel_size']=2
        params['strides']=1
        params['embedding_size']=3
        output = encoder(inputs,params)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={inputs:np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],dtype='float32')})
            self.assertTrue((output_val.shape == (batch_size,params['embedding_size'])), 'cnn_7 layers output')


if __name__ == '__main__':
    unittest.main()
