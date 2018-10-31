import numpy as np

from model.voice_dataset import get_input_function


def get_registerations(embeddings, label_ids, average_embedding=True):
    '''
    Group the embeddings by their labels, since one label may have multiple embeddings.
    :param embeddings: numpy ndarray,
    :param label_ids:label ids of each embedding
    :param average_embedding: bool, True to use the average of all the embeddings to represent an label
        False to use the list of all the embeddings to represent an label
    :return: dict of representative embeddings for each label
    '''
    # each person has a list of embeddings as his/her registeration
    registerations = dict()

    for embedding, label_id in zip(embeddings, label_ids):
        if label_id in registerations:
            registerations[label_id].append(embedding)
        else:
            registerations[label_id] = [embedding]
    if average_embedding:
        registerations_average=dict()
        for label_id in registerations:
            cur_embeddings = registerations[label_id]
            average_embedding = sum(cur_embeddings)/len(cur_embeddings)
            registerations_average[label_id]=[average_embedding]
        return registerations_average
    else:
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
                   **kwargs):
    label_ids = [-1 for _ in wav_files]

    predict_input_fn = lambda: get_input_function(
        wav_files,
        label_ids,
        is_training=False,
        **kwargs
    )

    embeddings = []
    for prediction in model.predict(predict_input_fn, yield_single_examples=False):
        embeddings.extend(prediction['embeddings'])

    return embeddings


def get_enrollments(enrollment_config):
    with open(enrollment_config) as f:
        enrollments = f.read().splitlines()
    return enrollments
