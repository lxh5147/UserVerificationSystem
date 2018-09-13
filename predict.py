import tensorflow as tf
from model.voice_dataset import input_fn, get_file_and_labels, from_ms_to_samples
from model.model_fn import create_model
import argparse
import sys
import os
import numpy as np
from model.voice_dataset import read_audio, extract_audio_feature


def l2_norm(embeddings):
    return embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)


def get_registerations(embeddings, label_ids):
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


def get_max_sim(embedding_unknown, embeddings_target):
    sim_max = -1.
    for embedding_target in embeddings_target:
        sim = _sim(embedding_unknown, embedding_target)
        if sim > sim_max:
            sim_max = sim
    return sim_max


def get_max_sim_and_id(embedding_unknown, embeddings_registered):
    # find the max similarity between the unknown embeddings and all the registered embeddings
    sim_max = -1.
    id_max = -1
    for id in embeddings_registered:
        sim = get_max_sim(embedding_unknown, embeddings_registered[id])
        if sim > sim_max:
            sim_max = sim
            id_max = id
    return sim_max, id_max


def get_embeddings(model,
                   wav_files,
                   desired_ms,
                   window_size_ms,
                   window_stride_ms,
                   sample_rate,
                   magnitude_squared,
                   dct_coefficient_count,
                   batch_size):
    desired_samples = from_ms_to_samples(sample_rate, desired_ms)
    window_size_samples = from_ms_to_samples(sample_rate, window_size_ms)
    window_stride_samples = from_ms_to_samples(sample_rate, window_stride_ms)
    label_ids = [-1 for _ in wav_files]
    predict_input_fn = lambda: input_fn(
        wav_files=wav_files,
        labels=label_ids,
        desired_samples=desired_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        magnitude_squared=magnitude_squared,
        dct_coefficient_count=dct_coefficient_count,
        batch_size=batch_size,
        is_training=False
    )

    embeddings = []
    for prediction in model.predict(predict_input_fn, yield_single_examples=False):
        embeddings.extend(prediction['embeddings'])

    embeddings_normed = l2_norm(embeddings)
    return embeddings_normed
