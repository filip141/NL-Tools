import numpy as np


def crossvalidate(n_groups, training_set):
    cross_groups = []
    length_diff = len(training_set) / n_groups
    for token_idx in xrange(1, n_groups + 1):
        start_tok = (token_idx - 1) * length_diff
        end_tok = token_idx * length_diff
        cross_groups.append(training_set[start_tok: end_tok])
    return cross_groups


def split_dataset(percent, dataset):
    data_len = len(dataset)
    test_len = int(percent * data_len)
    data_end = data_len - test_len
    return dataset[:data_end], dataset[data_end:]
