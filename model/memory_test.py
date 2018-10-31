import unittest

import numpy as np
import tensorflow as tf

from model.memory import read_memory


class MyTestCase(unittest.TestCase):
    def test_read_memory(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=(2, 3))
        output = read_memory(inputs)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={inputs: \
                                                         np.asarray([[1, 2, 3], [4, 5, 6]],
                                                                    dtype='float32')})
            self.assertTrue((output_val == np.asarray([[0, 0, 0], [0, 0, 0]])).all(), 'output')


if __name__ == '__main__':
    unittest.main()
