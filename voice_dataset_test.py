import tensorflow as tf
from model.voice_dataset import input_fn,get_labels,get_wav_files,from_ms_to_samples,dataset
def test_get_dataset_first():

    wav_files = [r'./data/train/121624931534904112937-0.wav',
                 r'./data/train/121624931534904112937-1.wav',
                 r'./data/train/121624931534904112937-2.wav'
                 ]
    labels=[0,1,2]
    desired_samples = 1600
    window_size_samples = 400
    window_stride_samples = 100
    voice_dataset = dataset(wav_files,
                            labels,
                            desired_samples,
                            window_size_samples,
                            window_stride_samples,
                            )
    features, labels = voice_dataset.make_one_shot_iterator().get_next()
    return features, labels
def main():
    with tf.Session() as sess:
        for i in range(10):
            features, labels = sess.run(test_get_dataset_first())
            print("labels:{}".format(labels))
main()

