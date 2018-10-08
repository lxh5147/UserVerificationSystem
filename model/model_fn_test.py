import unittest
import tensorflow as tf
import numpy as np
from model.model_fn import _get_encoder,model_fn,create_model

class ModelfnTestCase(unittest.TestCase):
    def test_get_encoder(self):
        name='cnn'
        encoder_cnn=_get_encoder(name)
        batch_size = 2
        inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2, 3))
        params = {'filters': '64',
                 'blocks': 1,
                 'kernel_size': 2,
                 'strides': 1,
                 'embedding_size': 3
                 }
        output = encoder_cnn(inputs, params)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={inputs: \
                                                         np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                                                                    dtype='float32')})
        self.assertTrue((output_val.shape == (batch_size,params['embedding_size'])),'test _get_encoder function call')




if __name__=='__main__':
    unittest.main()