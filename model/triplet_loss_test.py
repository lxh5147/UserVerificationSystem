import unittest
import tensorflow as tf
from model.triplet_loss import _pairwise_distances,_get_anchor_positive_triplet_mask,_get_anchor_negative_triplet_mask,_get_triplet_mask,\
batch_all_triplet_loss,batch_hard_triplet_loss
import numpy as np

class TripletlosssTestCase(unittest.TestCase):
    def test_pairwise_distances(self):
        embeddings = tf.placeholder(dtype=tf.float32, shape=(2,3))
        output=_pairwise_distances(embeddings)
        init_op=tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val=sess.run(output,feed_dict={embeddings:np.asarray([[1,2,3],[1,3,4]],dtype='float32')})
        self.assertTrue((output_val==np.asarray([[0.,1.41421342],[1.41421342,0.]],dtype='float32')).all(),'the value of output')
    def test__get_anchor_positive_triplet_mask(self):
        labels=tf.placeholder(dtype=tf.int32,shape=(4,))
        output=_get_anchor_positive_triplet_mask(labels)
        init_op=tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val=sess.run(output,feed_dict={labels:np.asarray([0,1,2,3])})
        self.assertTrue((output_val==np.asarray([[False, False, False, False],
                                                 [False, False, False, False],
                                                 [False, False, False, False],
                                                 [False, False, False, False]])).all(),'get anchor positive triplet mask')
    def test_get_anchor_negative_triplet_mask(self):
        labels = tf.placeholder(dtype=tf.int32,shape=(4,))
        output=_get_anchor_negative_triplet_mask(labels)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val=sess.run(output,feed_dict={labels:np.asarray([0,1,2,3])})
        self.assertTrue((output_val==np.asarray([[False,  True,  True,  True],
                                                 [ True, False,  True,  True],
                                                 [ True,  True, False,  True],
                                                 [ True,  True,  True, False]])).all(),'get anchor negative triplet mask')
    def test_get_triplet_mask(self):
        labels=tf.placeholder(dtype=tf.int32,shape=(4,))
        output = _get_triplet_mask(labels)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val=sess.run(output,feed_dict={labels:np.asarray([0,1,2,3],dtype='int32')})
        self.assertTrue((output_val==np.asarray([[[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]],
               [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]],
               [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]],
               [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]]])).all(),'get triplet mask')
    def test_batch_all_triplet_loss(self):
        embeddings = tf.placeholder(dtype=tf.float32, shape=(2,3))
        labels = tf.placeholder(dtype=tf.int32, shape=(2,))
        margin=0.1
        output1,_=batch_all_triplet_loss(labels,embeddings,margin,squared=False)
        init_op=tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output1,feed_dict={labels: np.asarray([0, 1], dtype='int32'),\
                                                     embeddings: np.asarray([[1,5,3],[2,3,4]], dtype='float32')})
            #print(output_val)
        self.assertTrue((output_val == 0.0),'the value of output')
    def test_batch_hard_triplet_loss(self):
        embeddings = tf.placeholder(dtype=tf.float32, shape=(2, 3))
        labels = tf.placeholder(dtype=tf.int32, shape=(2,))
        margin = 0.1
        output = batch_hard_triplet_loss(labels, embeddings, margin,squared=False)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            output_val = sess.run(output, feed_dict={labels: np.asarray([2, 1], dtype='int32'),embeddings:np.asarray([[1,2,3],[1,3,4]],dtype='float32')})
        self.assertTrue((output_val == 0.0).all(),'the value of output')



if __name__=='__main__':
    unittest.main()





