import argparse
import base64
import datetime
import json
import os
import sys
import uuid
import wave
from shutil import move, rmtree
from wsgiref.simple_server import make_server

import numpy as np
import tensorflow as tf

from model.model_fn import create_model
from predict import get_embeddings, get_enrollments, get_max_sim_and_id

# function result
ACCEPT = 0
REJECT_BELOW_THRESHOLD = -1
REJECT_CONFUSED = -2
REJECT_NOT_EXIST = -3
# function supported
FUNC_DELETE = '1'
FUNC_ENROLL = '2'
FUNC_VERIFY = '3'
FUNC_IDENTIFY = '4'
FUNC_COLLECT_DATA_ENROLLMENT = '8'
FUNC_COLLECT_DATA_VERIFICATION = '9'
FUNC_COLLECT_DATA_IDENTIFICATION = '9'


def _write_pcm16_wav(output_file, audio):
    '''
    write int16 audio to a wav file
    :param output_file: the wav file to write
    :param audio: the audio stream, type numpy int16 array
    :return: None
    '''
    with wave.open(output_file, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(FLAGS.sample_rate)
        writer.writeframes(audio)


def _parse_environ(environ):
    '''
    extract parameters from the environment. The payload is in the wsgi.input field as a binary data stream.
    :param environ: the environment from which to extract parameters.
    :return: device_id, the id of the device that calls this service
            user_id, string,the id of the user to be verified or claimed
            func_id, string,the id of the function asked to be executed
            streams, a list of binary streams sent from the device
    '''
    request_body_encoded = environ['wsgi.input'].read(int(environ.get('CONTENT_LENGTH', 0)))
    request_body = json.loads(request_body_encoded.decode())
    device_id = request_body.get('device_id', '')
    user_id = request_body.get('user_id', '')
    func_id = request_body.get('func_id', '')
    streams_encoded = request_body.get('streams', [])
    streams = []
    for stream in streams_encoded:
        streams.append(base64.b64decode(stream))
    return device_id, user_id, func_id, streams


def _get_device_root_path(device_id):
    '''
    Each device has a root path that stores all the data to server this device, such as users registered
     for this device, data logged for verification and identification.
    :param device_id: string, the device id
    :return: the root path of the device
    '''
    return os.path.join(FLAGS.data_dir, '__device_' + device_id)


def _get_user_root_path(device_id, user_id):
    '''
    Each registered user in a device has one root path that stores all the data to serve this user, e.g., enrollment
    data.
    :param device_id: string, device id
    :param user_id: string, user id
    :return: the root path of the user
    '''
    return os.path.join(_get_device_root_path(device_id), '__user_' + user_id)


def _get_user_ids(device_id):
    user_ids = []
    for name in os.listdir(os.path.join(FLAGS.data_dir, device_id)):
        if name.startswith('__user_'):
            user_ids.append(name[len('__user_'):])
    return user_ids


def _ensure_user_root_path(device_id, user_id):
    user_root_path = _get_user_root_path(device_id, user_id)
    if not os.path.exists(user_root_path):
        os.makedirs(user_root_path)
    return user_root_path


def _delete_user(device_id, user_id):
    device_root_path = _get_device_root_path(device_id)
    user_root_path = _get_user_root_path(device_id, user_id)
    # if the folder is already in the __DELETE__ folder, we delete the old one before moving
    if os.exists(os.path.join(device_root_path, '__deleted__', '__user_' + user_id)):
        rmtree()
    # move to the __delete__ folder
    move(user_root_path, os.path.join(device_root_path, '__deleted__'))


def _save_pcm_stream_to_wav(path, stream):
    uniq_filename = uuid.uuid4().hex[:6].upper() + '.wav'
    output_file = os.path.join(path, uniq_filename)
    _write_pcm16_wav(output_file, stream, sample_rate=FLAGS.sample_rate)
    return output_file, uniq_filename


def _get_enrollment_filenames(device_id, user_id):
    user_root_path = _get_user_root_path(device_id, user_id)
    enrollment_config = os.path.join(user_root_path, 'enrollment_config')
    if os.path.exists(enrollment_config):
        return get_enrollments(enrollment_config)
    else:
        return []


def _save_enrollment_config(device_id, user_id, files):
    user_root_path = _get_user_root_path(device_id, user_id)
    enrollment_config = os.path.join(user_root_path, 'enrollment_config')
    with open(enrollment_config, 'w') as fw:
        fw.writelines(files)


def _enroll(model, device_id, user_id, streams):
    _ensure_user_root_path(device_id, user_id)
    enrollment_filenames = _get_enrollment_filenames(device_id, user_id)
    files = []
    user_root_path = _get_user_root_path(device_id, user_id)
    for stream in streams:
        output_file, output_filename = _save_pcm_stream_to_wav(user_root_path, stream)
        files.append(output_file)
        enrollment_filenames.append(output_filename)
    # compute and save embeddings
    embeddings = get_embeddings(model,
                                files,
                                FLAGS.desired_ms,
                                FLAGS.window_size_ms,
                                FLAGS.window_stride_ms,
                                FLAGS.sample_rate,
                                FLAGS.magnitude_squared,
                                FLAGS.dct_coefficient_count,
                                FLAGS.batch_size)
    for i, file in enumerate(files):
        embedding_file = file + '.npy'
        np.save(embedding_file, embeddings[i])
    # update config
    _save_enrollment_config(device_id, user_id, enrollment_filenames)


def _load_registerations(device_id):
    user_ids = _get_user_ids(device_id)
    registerations = dict()
    for user_id in user_ids:
        files = _get_enrollment_filenames(device_id, user_id)
        user_root_path = _get_user_root_path(device_id, user_id)
        embeddings = []
        for i, file in enumerate(files):
            embedding_file = os.path.join(user_root_path, file + '.npy')
            embedding = np.load(embedding_file)
            embeddings.append(embedding)
        registerations[user_id] = embeddings
    return registerations


def _verify(embedding_unknown, device_id, claimed_user_id):
    if device_id not in grouped_registerations:
        registerations = _load_registerations(device_id)
        grouped_registerations[device_id] = registerations
    else:
        registerations = grouped_registerations[device_id]
    if claimed_user_id not in registerations:
        return REJECT_NOT_EXIST, -1
    sim_max, id_max = get_max_sim_and_id(embedding_unknown, registerations)
    if id_max == claimed_user_id:
        if sim_max >= FLAGS.threshold:
            return ACCEPT, sim_max
        else:
            return REJECT_BELOW_THRESHOLD, sim_max
    else:
        return REJECT_CONFUSED, sim_max


def _identification(embedding_unknown, device_id):
    if device_id not in grouped_registerations:
        registerations = _load_registerations(device_id)
        grouped_registerations[device_id] = registerations
    else:
        registerations = grouped_registerations[device_id]
    sim_max, id_max = get_max_sim_and_id(embedding_unknown, registerations)
    if sim_max >= FLAGS.threshold:
        return ACCEPT, id_max, sim_max
    else:
        return REJECT_BELOW_THRESHOLD, id_max, sim_max


def _get_embedding(model, filepath):
    embeddings = get_embeddings(model,
                                [filepath],
                                batch_size=1,
                                **FLAGS.__dict__)
    return embeddings[0]


def _get_data_collection_root_path(device_id, user_id):
    return os.path.join(FLAGS.data_dir, '__data_collected__' '__device_' + device_id, '__user_' + user_id)


def _application(environ, start_response):
    method = environ['REQUEST_METHOD']
    path = environ['PATH_INFO']
    start_response('200 OK', [('Content-Type', 'application/json')])
    device_id, user_id, function_id, streams = _parse_environ(environ)
    result = dict()

    tf.logging.info(
        "Request with method:{}, path:{}, device_id:{}, user_id:{}, function_id:{}, num_streams:{}".format( \
            method, path, device_id, user_id, function_id, len(streams) if streams else 'None'))

    start = datetime.datetime.now()

    # enroll
    if function_id == FUNC_ENROLL:
        _enroll(device_id,
                user_id,
                streams
                )
        # refresh the registerations
        if device_id in grouped_registerations:
            grouped_registerations[device_id] = _load_registerations(device_id)

    # delete
    if function_id == FUNC_DELETE:
        _delete_user(device_id, user_id)
        if device_id in grouped_registerations:
            grouped_registerations[device_id] = _load_registerations(device_id)

    # verify
    if function_id == FUNC_VERIFY:
        assert (len(streams) == 1)
        device_root_path = _get_device_root_path(device_id)
        output_file = _save_pcm_stream_to_wav(os.path.join(device_root_path, '__verification__'), streams[0])
        embedding_unknown = _get_embedding(model, output_file)
        status_code, sim = _verify(embedding_unknown, grouped_registerations, device_id, user_id)
        result['status_code'] = status_code
        result['sim_score'] = sim

    # identification
    if function_id == FUNC_IDENTIFY:
        assert (len(streams) == 1)
        device_root_path = _get_device_root_path(device_id)
        output_file = _save_pcm_stream_to_wav(os.path.join(device_root_path, '__identification__'), streams[0])
        embedding_unknown = _get_embedding(model, output_file)
        status_code, target_user_id, sim = _identification(embedding_unknown, grouped_registerations, device_id)
        result['status_code'] = status_code
        result['user_id'] = target_user_id
        result['sim_score'] = sim

    # collect data
    if function_id in [FUNC_COLLECT_DATA_ENROLLMENT, FUNC_COLLECT_DATA_VERIFICATION, FUNC_COLLECT_DATA_IDENTIFICATION]:
        data_path_root = os.path.join(_get_data_collection_root_path(device_id, user_id), function_id)
        if not os.path.exists(data_path_root):
            os.makedirs(data_path_root)
        for stream in streams:
            _save_pcm_stream_to_wav(data_path_root, stream)

    end = datetime.datetime.now()
    elapsed = end - start
    tf.logging.info("Response with result:{}, elapsed:{}".format(result, elapsed))

    return [json.dumps(result)]


def main(_):
    global model
    global grouped_registerations
    tf.logging.set_verbosity(tf.logging.INFO)
    model = create_model(
        model_dir=FLAGS.model_dir,
        params={
            **FLAGS.__dict__
        })
    grouped_registerations = dict()
    httpd = make_server(host=FLAGS.host,
                        port=FLAGS.port,
                        app=_application
                        )
    tf.logging.info("serving http on port {0}...".format(FLAGS.port))
    httpd.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./tmp_model',
        help='model dir')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/online',
        help='data dir')
    parser.add_argument(
        '--input_feature',
        type=str,
        default='fbank',
        help='Input feature: Use raw|mfcc|fbank|logfbank. Only raw is valid if the encoder is sinc_*')
    parser.add_argument(
        '--normalize_frames',
        type=bool,
        default=False,
        help='If the features should be normalized across all frames for each dimension')
    parser.add_argument(
        '--encoder',
        type=str,
        default='rescnn',
        help='Encoder that encodes a wav to a vector. Use cnn|resnet|sinc_cnn|sinc_resnet')
    parser.add_argument(
        '--sinc_freq_scale',
        type=float,
        default=16000.,
        help='Frequency scale of sinc input feature extractor')
    parser.add_argument(
        '--sinc_kernel_size',
        type=int,
        default=3,
        help='Kernel size of sinc input feature extractor')
    parser.add_argument(
        '--filters',
        type=int,
        nargs='+',
        default=[64, 128, 256, 512],
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
        default=512,
        help='embedding_size')
    # if memory_cells > 0, the memory network will be enabled, and the output will be the weighted memory cells.
    parser.add_argument(
        '--memory_cells',
        type=int,
        default=0,
        help='number of memory cells.')
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
        '--input_feature_dim',
        type=int,
        default=64,
        help='Dimension of input feature')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch_size')
    parser.add_argument(
        '--threshold',
        type=float,
        default=-1.,
        help='If the similarity between two wav files is no less than this threshold, they are considered from the same person.')

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + _)
