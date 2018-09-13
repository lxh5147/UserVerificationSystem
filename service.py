from wsgiref.simple_server import make_server
import tensorflow as tf
from model.model_fn import create_model
import argparse
import sys
import os
from predict import get_registerations, get_max_sim, get_max_sim_and_id, get_embeddings
import json
import base64
import wave
from shutil import rmtree
import uuid

FLAGS = None


def _write_pcm16_wav(output_file, audio, sample_rate=16000):
    with wave.open(output_file, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(audio)

def _parse_environ(environ):
    request_body_encoded = environ['wsgi.input'].read(int(environ.get('CONTENT_LENGTH', 0)))
    request_body = json.loads(request_body_encoded.decode())
    device_id = request_body.get('family_id', '')
    speaker_id = request_body.get('user_id', '')
    function_id = request_body.get('func', '')
    streams_encoded = request_body.get('streams', None)
    streams = []
    for stream in streams_encoded:
        streams.append(base64.b64decode(stream))
    return device_id, speaker_id, function_id, streams


def _get_user_root_path(device_id, user_id):
    return os.path.join(FLAGS.data_dir, device_id, user_id)


def _ensure_user_root_path(device_id, user_id):
    user_root_path = _get_user_root_path(device_id, user_id)
    if not os.path.exists(user_root_path):
        os.mkdir(user_root_path)
    return user_root_path


def _delete_user(device_id, user_id):
    user_root_path = _get_user_root_path(device_id, user_id)
    if os.path.exists(user_root_path):
        rmtree(user_root_path)

def _save_pcm_stream(device_id, user_id, stream):
    user_root_path = _get_user_root_path(device_id, user_id)
    uniq_filename = uuid.uuid4().hex[:6].upper()
    output_filepath = os.path.join(user_root_path,uniq_filename +'.wav')
    _write_pcm16_wav(output_filepath, stream, sample_rate=16000)

def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    filters = map(lambda _: int(_), FLAGS.filters.split(','))
    model = create_model(
        model_dir=FLAGS.model_dir,
        params={
            'filters': filters,
            'blocks': FLAGS.blocks,
            'kernel_size': FLAGS.kernel_size,
            'strides': FLAGS.strides,
            'embedding_size': FLAGS.embedding_size,
            'encoder': FLAGS.encoder
        })

    def application(environ, start_response):
        method = environ['REQUEST_METHOD']
        path = environ['PATH_INFO']
        start_response('200 OK', [('Content-Type', 'application/json')])
        device_id, speaker_id, function_id, streams = _parse_environ(environ)
        pass

    httpd = make_server(host=FLAGS.host,
                        port=FLAGS.port,
                        app=application
                        )

    tf.logging.info("serving http on port {0}...".format(FLAGS.port))
    httpd.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./tmp_model',
        help='model_dir')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='data dir')
    parser.add_argument(
        '--encoder',
        type=str,
        default='cnn',
        help='Encoder that encodes a wav to a vector. Use cnn or resnet')
    parser.add_argument(
        '--filters',
        type=str,
        default='64,128,256,512',
        help='filters')
    parser.add_argument(
        '--blocks',
        type=int,
        default=3,
        help='blocks')
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
        help='kernel_size')
    parser.add_argument(
        '--strides',
        type=int,
        default=2,
        help='strides of conv')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='embedding_size')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Sample rate of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.', )
    parser.add_argument(
        '--desired_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.')
    parser.add_argument(
        '--magnitude_squared',
        type=bool,
        default=True,
        help='magnitude_squared')
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='dct_coefficient_count')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch_size')
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='If the similarity between two wav files is no less than this threshold, they are considered from the same person.')

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + _)
