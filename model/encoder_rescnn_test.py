import unittest

import numpy as np
import tensorflow as tf

from model.encoder_rescnn import encoder
from model.triplet_loss import batch_all_triplet_loss


class EncoderTestCase(unittest.TestCase):
    def test_encoder(self):
        tf.set_random_seed(2008)
        params = {
            'embedding_size': 3,
            'input_feature_dim': 3,
            'l2_regularization_weight': 0.0001,
            'margin': 0.2,
            'squared': True
        }
        # bs, ts, dim
        inputs = tf.placeholder(shape=(None, 2, 1), dtype=tf.float32)
        labels = tf.placeholder(shape=(None,), dtype=tf.int64)
        embeddings = encoder(inputs,
                             params,
                             is_training=True)

        loss_triplet, _ = batch_all_triplet_loss(labels,
                                                 embeddings,
                                                 margin=params['margin'],
                                                 squared=params['squared'])
        loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            inputs_val = np.asarray([[[1], [2]], [[3], [4]], [[5], [6]]], dtype='float32')
            labels_val = np.asarray([1,2,2], dtype='int64')
            embeddings_val, loss_triplet_val, loss_reg_val = sess.run([embeddings,loss_triplet,loss_reg], feed_dict={inputs:inputs_val, labels: labels_val} )

            print(embeddings_val)
            print(loss_triplet_val)
            print(loss_reg_val)

if __name__ == '__main__':
    unittest.main()
