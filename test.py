import tensorflow as tf
import numpy as np
from model.voice_dataset import read_audio



def test_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    dataset = dataset.shuffle(3)
    dataset = dataset.repeat().batch(2)
    iterator = dataset.make_one_shot_iterator()
    one_batch = iterator.get_next()
    i=0
    with tf.Session() as sess:
        while i<8:
            print(sess.run(one_batch))
            i += 1

test_dataset()