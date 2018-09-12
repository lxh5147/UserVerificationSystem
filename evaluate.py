import tensorflow as tf
from model.voice_dataset import input_fn, get_file_and_labels, from_ms_to_samples
from model.model_fn import create_model
import argparse
import sys
import os
import numpy as np

FLAGS = None


def _l2_norm(embeddings):
    return embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)


def _get_registerations(embeddings, label_ids):
    # each person has a list of embeddings as his/her registeration
    registerations = dict()

    for embedding, label_id in zip(embeddings, label_ids):
        if label_id in registerations:
            registerations[label_id].append(embedding)
        else:
            registerations[label_id] = [embedding]
    return registerations


def _sim(embedding_1, embedding_2):
    return np.dot(embedding_1, embedding_2)


def _get_max_sim(embedding_unknown, embeddings_target):
    sim_max = -1.
    for embedding_target in embeddings_target:
        sim = _sim(embedding_unknown, embedding_target)
        if sim > sim_max:
            sim_max = sim
    return sim_max


def _get_max_sim_and_id(embedding_unknown, embeddings_registered):
    # find the max similarity between the unknown embeddings and all the registered embeddings
    sim_max = -1.
    id_max = -1
    for id in embeddings_registered:
        sim = _get_max_sim(embedding_unknown, embeddings_registered[id])
        if sim > sim_max:
            sim_max = sim
            id_max = id
    return sim_max, id_max


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


def _identification_correct(to_be_identified, sims, label_ids):
    correct = []
    for i, j in enumerate(sims):
        embedding_index = to_be_identified[i][0]
        true_id = label_ids[embedding_index]
        sim, id = j
        if true_id == id:
            correct.append(embedding_index)
    return correct


def _eer(to_be_verified, verification_sim, true_v_a, true_v_r):
    fa_rates = []
    fr_rates = []
    gap = []
    for threshold in [0.01 * i - 1.0 for i in range(200)]:
        fa, fr = _verfication_fa_fr(to_be_verified, verification_sim, true_v_a, true_v_r, threshold)
        fa_rate = len(fa) / len(true_v_r)
        fr_rate = len(fr) / len(true_v_a)
        fa_rates.append(fa_rate)
        fr_rates.append(fr_rate)
        gap.append(abs(fa_rate - fr_rate))

    min_pos = gap.index(min(gap))
    eer = (fa_rates[min_pos] + fr_rates[min_pos]) / 2
    eer_threshold = 0.01 * min_pos - 1.0
    return eer, eer_threshold


def _evaluate_verification(embeddings, label_ids, registerations, to_be_verified, threshold=None):
    verification_sim = []
    true_v_a = []  # true accept
    true_v_r = []  # true reject

    for embedding_index, claim_id in to_be_verified:
        embeddings_target = registerations[claim_id]
        embedding_unknown = embeddings[embedding_index]
        sim = _get_max_sim(embedding_unknown, embeddings_target)
        verification_sim.append(sim)
        true_id = label_ids[embedding_index]
        if true_id == claim_id:
            true_v_a.append(embedding_index)
        else:
            true_v_r.append(embedding_index)
    if threshold:
        fa, fr = _verfication_fa_fr(to_be_verified, verification_sim, true_v_a, true_v_r, threshold)
        fa_rate = len(fa) / len(true_v_r)
        fr_rate = len(fr) / len(true_v_a)
        return fa_rate, fr_rate, threshold
    else:
        # verification performance
        eer, eer_thredhold = _eer(to_be_verified, verification_sim, true_v_a, true_v_r)
        return eer, eer, eer_thredhold


def _evaluate_identification(embeddings, label_ids, registerations, to_be_identified, groups):
    grouped_registerations = dict()
    identification_sim = []

    for embedding_index, target_group_id in to_be_identified:
        if target_group_id: # not empty
            if target_group_id in grouped_registerations:
                target_registerations = grouped_registerations[target_group_id]
            else:
                target_registerations = dict()
                group = groups[target_group_id]
                for id in group:
                    target_registerations[id] = registerations[id]
                grouped_registerations[target_group_id] = target_registerations
        else:
            target_registerations = registerations
        sim, id = _get_max_sim_and_id(embeddings[embedding_index], target_registerations)
        identification_sim.append((sim, id))

    # identification performance
    correct_identified = _identification_correct(to_be_identified, identification_sim, label_ids)
    acc = len(correct_identified) / len(to_be_identified)
    return acc


def evaluate(embeddings, label_ids, enrollments, to_be_verified, to_be_identified, groups, threshold=None):
    embeddings_normed = _l2_norm(embeddings)
    registerations = _get_registerations([embeddings_normed[i] for i in enrollments],
                                         [label_ids[i] for i in enrollments])
    fa_rate, fr_rate, threshold = _evaluate_verification(embeddings, label_ids, registerations, to_be_verified,
                                                         threshold)
    acc = _evaluate_identification(embeddings_normed, label_ids, registerations, to_be_identified, groups)
    return fa_rate, fr_rate, threshold, acc


'''
eval_folder: 
    wav sub directory: wav files
    labels file
    enrollment_config: wav_file_id
    verfication_config file: wav_file_id,claimed_label # 121624931534904112937-0.wav --> 121624931534904112937-0
    identification_config file: wav_file_id,group_id #group_id < 0 if we consider the group with all users
    group_config file: group id,label of group_member 1,label of group_member 2,...
'''


def _get_enrollments(enrollment_config):
    with open(enrollment_config) as f:
        enrollments = f.read().splitlines()
    return enrollments


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
    enrollments = _get_enrollments(os.path.join(FLAGS.data_dir, 'enrollment_config'))
    to_be_verified = _get_to_be_verified(os.path.join(FLAGS.data_dir, 'verification_config'))
    to_be_identified = _get_to_be_identified(os.path.join(FLAGS.data_dir, 'identification_config'))
    wav_file_id_to_index = _get_file_id_to_index(wav_files)
    # TODO validate configurations
    # transform the configurations: wav file id --> index, label_id --> label_index
    groups_transformed = dict()
    for group_id in groups:
        group = [label_to_id[i] for i in groups[group_id]]
        groups_transformed[group_id] = group
    groups = groups_transformed

    enrollments = [wav_file_id_to_index[i] for i in enrollments]
    to_be_verified = [(wav_file_id_to_index[i], label_to_id[j]) for i, j in to_be_verified]
    to_be_identified = [(wav_file_id_to_index[i], group_id) for i, group_id in to_be_identified]

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

    desired_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.desired_ms)
    window_size_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_size_ms)
    window_stride_samples = from_ms_to_samples(FLAGS.sample_rate, FLAGS.window_stride_ms)
    eval_input_fn = lambda: input_fn(
        wav_files=wav_files,
        labels=label_ids,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=FLAGS.magnitude_squared,
        dct_coefficient_count=FLAGS.dct_coefficient_count,
        batch_size=FLAGS.batch_size,
        is_training=False
    )

    all_embeddings = []
    for prediction in model.predict(eval_input_fn, yield_single_examples=False):
        all_embeddings.extend(prediction['embeddings'])

    fa_rate, fr_rate, threshold, acc = evaluate(np.asanyarray(all_embeddings),
                                                label_ids,
                                                enrollments,
                                                to_be_verified,
                                                to_be_identified,
                                                groups,
                                                FLAGS.threshold)
    eval_msg_template = 'verfication false accept rate:{}\n' + \
                        '            false reject rate:{}\n' + \
                        '            threshold:{}\n' + \
                        'identification accuracy:{}'

    tf.logging.info(eval_msg_template.format(fa_rate, fr_rate, threshold, acc))


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
        help='model_dir')
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
