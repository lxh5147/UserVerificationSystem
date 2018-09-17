import unittest

import numpy as np
import tensorflow as tf

from model.attention import attention


class AttentionTestCase(unittest.TestCase):
    def test_attention(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=(2, 2, 3))
        output = attention(inputs)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={inputs: \
                                                         np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                                                                    dtype='float32')})
            self.assertTrue((output_val == np.asarray([[2.5, 3.5, 4.5], [8.5, 9.5, 10.5]])).all(), 'output')


if __name__ == '__main__':
    unittest.main()
