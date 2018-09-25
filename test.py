import tensorflow as tf



from  model.voice_dataset import _create_generator

wav_files =['1.wav','2.wav','3.wav','4.wav']
labels=[1,2,3,1]

ds = tf.data.Dataset.from_generator(_create_generator(wav_files,labels),(tf.string, tf.int64),\
                               (tf.TensorShape([]), tf.TensorShape([])))

ds = ds.repeat()

value = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    for i in range(16):
        if i % len(labels) == 0:
            print('==')
        print(sess.run(value))

