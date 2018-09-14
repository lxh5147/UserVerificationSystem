import argparse
import os
import sys

import tensorflow as tf

from model.model_fn import create_model
from model.voice_dataset import get_file_and_labels
from predict import get_registerations, get_max_sim, get_max_sim_and_id, get_embeddings, get_enrollments


def _verfication_fa_fr(to_be_verified, sims, true_a, true_r, threshold=0.7):
    # return the indexes false rejected and false accepted
    fa = []  # false accept
    fr = []  # false reject
    for i, sim in enumerate(sims):
        embedding_index = to_be_verified[i][0]
        if sim >= threshold:  # we believe we should accept
            if embedding_index in true_r:
                fa.append(embedding_index)
        else:  # we believe we should reject
            if embedding_index in true_a:
                fr.append(embedding_index)
    return fa, fr


def _eer(fa_rates, fr_rates):
    gap = [abs(fa_rate - fr_rate) for fa_rate, fr_rate in zip(fa_rates, fr_rates)]
    min_pos = gap.index(min(gap))
    eer = (fa_rates[min_pos] + fr_rates[min_pos]) / 2
    eer_threshold = 0.01 * min_pos - 1.0
    return eer, eer_threshold


def _verification_eer(to_be_verified, verification_sim, true_a, true_r):
    fa_rates = []
    fr_rates = []
    for threshold in [0.01 * i - 1.0 for i in range(200)]:
        fa, fr = _verfication_fa_fr(to_be_verified, verification_sim, true_a, true_r, threshold)
        fa_rate = len(fa) / len(true_r)
        fr_rate = len(fr) / len(true_a)
        fa_rates.append(fa_rate)
        fr_rates.append(fr_rate)
    return _eer(fa_rates, fr_rates)


def _evaluate_verification(embeddings, label_ids, registerations, to_be_verified, threshold=None):
    verification_sim = []
    true_a = []  # true accept
    true_r = []  # true reject

    for embedding_index, claim_id in to_be_verified:
        embeddings_target = registerations[claim_id]
        embedding_unknown = embeddings[embedding_index]
        sim = get_max_sim(embedding_unknown, embeddings_target)
        verification_sim.append(sim)
        true_id = label_ids[embedding_index]
        if true_id == claim_id:
            true_a.append(embedding_index)
        else:
            true_r.append(embedding_index)
    if threshold:
        fa, fr = _verfication_fa_fr(to_be_verified, verification_sim, true_a, true_r, threshold)
        if true_r:
            fa_rate = len(fa) / len(true_r)
        else:
            fa_rate = 0
        if true_a:
            fr_rate = len(fr) / len(true_a)
        else:
            fr_rate = 0
        return fa_rate, fr_rate, threshold
    else:
        # verification performance
        eer, eer_thredhold = _verification_eer(to_be_verified, verification_sim, true_a, true_r)
        return eer, eer, eer_thredhold


def _identification_fa_fr(to_be_identified, sims, label_ids, threshold=0.7):
    # return the indexes false rejected and false accepted
    fa = []  # false accept
    fr = []  # false reject
    for i, j in enumerate(sims):
        embedding_index = to_be_identified[i][0]
        true_id = label_ids[embedding_index]
        sim, id = j
        if true_id == id:
            if sim < threshold:
                fr.append(embedding_index)
        else:
            if sim >= threshold:
                fa.append(embedding_index)

    return fa, fr


def _identification_eer(to_be_identified, sims, label_ids, true_a, true_r):
    fa_rates = []
    fr_rates = []
    for threshold in [0.01 * i - 1.0 for i in range(200)]:
        fa, fr = _verfication_fa_fr(to_be_identified, sims, label_ids, threshold)
        fa_rate = len(fa) / len(true_r)
        fr_rate = len(fr) / len(true_a)
        fa_rates.append(fa_rate)
        fr_rates.append(fr_rate)
    return _eer(fa_rates, fr_rates)


def _evaluate_identification(embeddings, label_ids, registerations, to_be_identified, groups, threshold=None):
    grouped_registerations = dict()
    sims = []

    true_a = []  # true accept
    true_r = []  # true reject

    for embedding_index, target_group_id in to_be_identified:
        if target_group_id:  # not empty
            if target_group_id in grouped_registerations:
                group_registerations = grouped_registerations[target_group_id]
            else:
                group_registerations = dict()
                group = groups[target_group_id]
                for id in group:
                    group_registerations[id] = registerations[id]
                grouped_registerations[target_group_id] = group_registerations
        else:
            group_registerations = registerations
        sim, id = get_max_sim_and_id(embeddings[embedding_index], group_registerations)
        sims.append((sim, id))

        true_id = label_ids[embedding_index]
        if true_id in group_registerations:
            true_a.append(embedding_index)
        else:
            true_r.append(embedding_index)

    if threshold:
        fa, fr = _identification_fa_fr(to_be_identified, sims, label_ids, threshold=0.7)
        if true_r:
            fa_rate = len(fa) / len(true_r)
        else:
            fa_rate = 0
        if true_a:
            fr_rate = len(fr) / len(true_a)
        else:
            fr_rate = 0
        return fa_rate, fr_rate, threshold
    else:
        eer, eer_thredhold = _identification_eer(to_be_identified, sims, label_ids, true_a, true_r)
        return eer, eer, eer_thredhold


'''
eval_folder: 
    wav sub directory: wav files
    labels file
    enrollment_config: wav_file_id
    verfication_config file: wav_file_id,claimed_label # 121624931534904112937-0.wav --> 121624931534904112937-0
    identification_config file: wav_file_id,group_id #group_id < 0 if we consider the group with all users
    group_config file: group id,label of group_member 1,label of group_member 2,...
'''


def _get_groups(group_config_file):
    with open(group_config_file) as f:
        lines = f.read().splitlines()
    # map a line to an ID
    groups = dict()

    for line in lines:
        parts = line.split(',')
        group_id = parts[0]
        group_members = parts[1:]
        groups[group_id] = group_members

    return groups


def _get_to_be_verified(verfication_config_file):
    with open(verfication_config_file) as f:
        lines = f.read().splitlines()

    to_be_verified = []
    for line in lines:
        parts = line.split(',')
        wav_file = parts[0]
        claimed = parts[1]
        to_be_verified.append((wav_file, claimed))

    return to_be_verified


def _get_to_be_identified(identification_config_file):
    with open(identification_config_file) as f:
        lines = f.read().splitlines()

    to_be_identified = []

    for line in lines:
        parts = line.split(',')
        wav_file = parts[0]
        if len(parts) == 1:
            group_id = ''
        else:
            group_id = parts[1]

        to_be_identified.append((wav_file, group_id))

    return to_be_identified


def _get_file_id(file):
    file_id = os.path.basename(file)
    return file_id


def _get_file_id_to_index(files):
    file_id_to_index = dict()
    for i, file in enumerate(files):
        file_id = _get_file_id(file)
        file_id_to_index[file_id] = i
    return file_id_to_index


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the input function for training
    wav_files, label_ids, label_to_id = get_file_and_labels(os.path.join(FLAGS.data_dir, 'eval_labels'))
    wav_files = [os.path.join(FLAGS.data_dir, 'eval', wav_file) for wav_file in wav_files]

    groups = _get_groups(os.path.join(FLAGS.data_dir, 'groups_config'))
    enrollments = get_enrollments(os.path.join(FLAGS.data_dir, 'enrollment_config'))
    to_be_verified = _get_to_be_verified(os.path.join(FLAGS.data_dir, 'verification_config'))
    to_be_identified = _get_to_be_identified(os.path.join(FLAGS.data_dir, 'identification_config'))
    file_id_to_index = _get_file_id_to_index(wav_files)
    # TODO validate configurations
    # transform the configurations: wav file id --> index, label_id --> label_index
    groups_transformed = dict()
    for group_id in groups:
        group = [label_to_id[i] for i in groups[group_id]]
        groups_transformed[group_id] = group
    groups = groups_transformed

    enrollments = [file_id_to_index[i] for i in enrollments]
    to_be_verified = [(file_id_to_index[i], label_to_id[j]) for i, j in to_be_verified]
    to_be_identified = [(file_id_to_index[i], group_id) for i, group_id in to_be_identified]

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

    embeddings = get_embeddings(model,
                                wav_files=wav_files,
                                desired_ms=FLAGS.desired_ms,
                                window_size_ms=FLAGS.window_size_ms,
                                window_stride_ms=FLAGS.window_stride_ms,
                                sample_rate=FLAGS.sample_rate,
                                magnitude_squared=FLAGS.magnitude_squared,
                                dct_coefficient_count=FLAGS.dct_coefficient_count,
                                batch_size=FLAGS.batch_size)

    registerations = get_registerations([embeddings[i] for i in enrollments],
                                        [label_ids[i] for i in enrollments])
    fa_rate, fr_rate, threshold = _evaluate_verification(embeddings, label_ids, registerations, to_be_verified,
                                                         FLAGS.threshold)

    eval_msg_template = 'false accept rate:{}\n' + \
                        'false reject rate:{}\n' + \
                        'threshold:{}'

    tf.logging.info('verification performance')
    tf.logging.info(eval_msg_template.format(fa_rate, fr_rate, threshold))

    fa_rate, fr_rate, threshold = _evaluate_identification(embeddings, label_ids, registerations,
                                                           to_be_identified, groups, threshold)
    tf.logging.info('identification performance')
    tf.logging.info(eval_msg_template.format(fa_rate, fr_rate, threshold))


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
