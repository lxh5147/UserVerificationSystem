import unittest

import numpy as np
import tensorflow as tf

from model.cross_entropy_loss import cross_entropy_loss


class AttentionTestCase(unittest.TestCase):
    def test_attention(self):
        labels = tf.placeholder(dtype=tf.int32, shape=(3,))
        embeddings=tf.placeholder(dtype=tf.float32, shape=(3,2))
        num_classes=3
        output = cross_entropy_loss(labels, embeddings, num_classes)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={labels:np.asarray([0,1,2],dtype='int32'),embeddings:np.asarray([[0,1],[1,2],[2,3]],dtype='float32')})
            self.assertTrue((not output_val == 0), 'output')


if __name__ == '__main__':
    unittest.main()
