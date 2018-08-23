import tensorflow as tf

# 2*3
a = tf.constant([[1, 2, 3], [4, 5, 6]])
# 2*1
b = tf.constant([[1], [2]])

print(a * tf.expand_dims(tf.squeeze(b, -1), -1))


