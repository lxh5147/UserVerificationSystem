import argparse
import base64
import json
import os
import sys
import uuid
import wave
from shutil import rmtree
from wsgiref.simple_server import make_server

import numpy as np
import tensorflow as tf

from model.model_fn import create_model
from predict import get_embeddings, get_enrollments, get_max_sim_and_id

FLAGS = None


def _write_pcm16_wav(output_file, audio):
    with wave.open(output_file, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(FLAGS.sample_rate)
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
    return os.path.join(FLAGS.data_dir, device_id, '__user_' + user_id)

def _get_user_ids(device_id):
    user_ids =[]
    for name in os.listdir(os.path.join(FLAGS.data_dir, device_id)):
        if name.startswith('__user_'):
            user_ids.append(name[len('__user_'):])
    return user_ids

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
    output_file = os.path.join(user_root_path, uniq_filename + '.wav')
    _write_pcm16_wav(output_file, stream, sample_rate=16000)
    return output_file


def _get_enrollment_wav_files(device_id, user_id):
    user_root_path = _get_user_root_path(device_id, user_id)
    enrollment_config = os.path.join(user_root_path, 'enrollment_config')
    if os._exists(enrollment_config):
        return [os.path.join(user_root_path, i) for i in get_enrollments(enrollment_config)]
    else:
        return []


def _update_enrollment_config(device_id, user_id, wav_files):
    user_root_path = _get_user_root_path(device_id, user_id)
    enrollment_config = os.path.join(user_root_path, 'enrollment_config')
    with open(enrollment_config, 'w') as fw:
        fw.writelines(wav_files)


def _enroll_user(model,
                 device_id,
                 user_id,
                 streams
                ):
    _ensure_user_root_path(device_id, user_id)
    wav_files_exist = _get_enrollment_wav_files(device_id, user_id)
    wav_files = []
    for stream in streams:
        output_file = _save_pcm_stream(device_id, user_id, stream)
        wav_files.append(output_file)
    # compute and save embeddings
    embeddings = get_embeddings(model,
                                wav_files,
                                FLAGS.desired_ms,
                                FLAGS.window_size_ms,
                                FLAGS.window_stride_ms,
                                FLAGS.sample_rate,
                                FLAGS.magnitude_squared,
                                FLAGS.dct_coefficient_count,
                                FLAGS.batch_size)
    for i, wav_file in enumerate(wav_files):
        embedding_file = wav_file + '.npy'
        np.save(embedding_file, embeddings[i])
    # update config
    wav_files_exist.extend(wav_files)
    _update_enrollment_config(device_id, user_id, wav_files)


def _load_registerations(device_id):
    user_ids =_get_user_ids(device_id)
    registerations = dict()
    for user_id in user_ids:
        wav_files = _get_enrollment_wav_files(device_id, user_id)
        embeddings = []
        for i, wav_file in enumerate(wav_files):
            embedding_file = wav_file + '.npy'
            embedding = np.load(embedding_file)
            np.save(embedding_file, embeddings[i])
            embeddings.append(embedding)
        registerations[user_id]=embeddings

    return registerations

def _verify(embedding_unknown, grouped_registerations, device_id, claimed_user_id):
    if device_id not in grouped_registerations:
        registerations = _load_registerations(device_id)
        grouped_registerations[device_id]=registerations
    else:
        registerations = grouped_registerations[device_id]
    if claimed_user_id not in registerations:
        return False, -1, \
               'claimed user with user id {} in device {} is not registered'.format(claimed_user_id, device_id)


    sim_max, id_max = get_max_sim_and_id(embedding_unknown,registerations)
    if id_max == claimed_user_id:
        if sim_max >= FLAGS.threshold:
            return True, sim_max, ''
        else:
            return False, sim_max, \
                   'claimed user with user id {} in device {} is rejected since its similarity {} is less than the threshold {}'.format(claimed_user_id, device_id, sim_max, FLAGS.threshold)
    else:
        return False, sim_max,\
               'claimed user with user id {} in device {} is rejected since it is confused with the user {} with a similarity {}'.format(claimed_user_id, device_id, id_max, sim_max)


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
