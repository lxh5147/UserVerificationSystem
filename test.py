import tensorflow as tf

from model.voice_dataset import read_audio


def test_audio_too_short():
    desired_samples = 100000
    wav_file = tf.placeholder(dtype=tf.string)
    audio, sample_rate, all_samples = read_audio(wav_file,
                                                 desired_samples)
    with tf.Session() as sess:
        all_samples_val, audio_val = sess.run([all_samples, audio],
                                              feed_dict={wav_file: "./data/train/121624931534904112937-0.wav"})
        print("length:{} padded length:{}".format(all_samples_val, len(audio_val)))


test_audio_too_short()
