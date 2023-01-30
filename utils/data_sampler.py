import numpy as np
from numpy.lib.function_base import iterable
import torch
import sklearn.model_selection


def get_subset_indices(dataset: torch.utils.data.Dataset, n_train_samples: int,
                       random_seed: int):
    """
    Randomly selects n_train_samples number of indices from the dataset,
    stratified by dataset target labels.
    :param dataset: PyTorch train dataset
    :type dataset: torch.utils.data.Dataset
    :param n_train_samples: number of training samples to include in the
        training dataset
    :type n_train_samples: int
    :param random_seed: user-specified random seed for the selection of
        training examples
    :type random_seed: int
    :return: a (n_train_samples,) np.array containing the randomly-selected
        training indices
    """
    if len(dataset.samples) == n_train_samples:
        idx = np.arange(len(dataset))
        np.random.seed(random_seed)
        np.random.shuffle(idx)
    else:
        test_size = len(dataset.samples) - n_train_samples
        idx, _ = sklearn.model_selection.train_test_split(
            list(range(len(dataset.samples))),
            test_size=test_size,
            stratify=dataset.samples,
            random_state=random_seed, shuffle=True)
    return idx


def get_train_valid_indices(dataset: torch.utils.data.Dataset,
                            n_train_samples: int, n_valid_samples: int,
                            train_seed: int, valid_seed: int = 12345):
    """
    Randomly selects n_train_samples number of indices from the dataset,
    stratified by dataset target labels and n_valid_samples number of indices
    from the dataset, stratified by dataset target labels.

    :param dataset: PyTorch train dataset
    :type dataset: torch.utils.data.Dataset
    :param n_train_samples: number of training samples to include in the
        training dataset
    :type n_train_samples: int
    :param n_valid_samples: number of validation samples to include in the
        validation dataset
    :type n_valid_samples: int
    :param train_seed: user-specified random seed for the selection of
        training examples
    :type random_seed: int
    :param valid_seed: user-specified random seed for the selection of
        validation examples
    :type random_seed: int
    :return: a tuple of arrays containing the randomly-selected indices
    """
    learn_idx, valid_idx = train_test_split(
        list(range(len(dataset.labels))),
        test_size=n_valid_samples,
        stratify=dataset.labels,
        random_state=valid_seed,
        shuffle=True)

    train_idx, _ = train_test_split(
        learn_idx,
        test_size=len(learn_idx) - n_train_samples,
        stratify=np.array(dataset.labels)[learn_idx],
        random_state=train_seed,
        shuffle=True)

    # assert (np.count_nonzero(np.array(dataset.labels)[train_idx]) ==
    #         int(np.floor(n_train_samples / 2)))
    # assert (np.count_nonzero(np.array(dataset.labels)[valid_idx]) ==
    #         int(np.ceil(n_valid_samples / 2)))

    return train_idx, valid_idx


def train_test_split(idx: list, test_size: int, stratify: iterable,
                     random_state: int, shuffle: bool):
    """
    Performs a (stratified) train-test split similar to scikit-learn, except
    uses random shuffling to deterministically choose the same starting
    samples.

    Note: only works for binary classification for now.
    """

    np.random.seed(random_state)

    train_size = len(idx) - test_size

    if stratify is not None:
        pos_idx = np.argwhere(stratify).flatten().astype(int)
        neg_idx = np.argwhere(np.logical_not(stratify)).flatten().astype(int)
        pos_idx = np.array(idx)[pos_idx]
        neg_idx = np.array(idx)[neg_idx]
        assert len(pos_idx) + len(neg_idx) == len(idx)

        n_pos = int(np.floor(train_size / 2))
        n_neg = int(np.ceil(train_size / 2))
        assert abs(n_pos - n_neg) < 2

        if shuffle:
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
        train_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
        test_idx = np.concatenate([pos_idx[n_pos:], neg_idx[n_neg:]])
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
    else:
        if shuffle:
            np.random.shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]

    return train_idx, test_idx
