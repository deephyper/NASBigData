"""
https://github.com/tmatha/lstm
"""

import os
import logging
import numpy as np
import time
import math
import datetime
import collections

from scipy.sparse import *

HERE = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)


def features_labels(data_array, batch_size, seq_len, batch_first=True):
    """Splits the sequential data into batch_size number of sub_sequences and
  folds them into the requisite shape. This procedure is applied to the data to
  derive the features array. This procedure is repeated to derive the labels
  array also, except in this case the data is shifted by one time step.
  Returns a named tuple of features and labels.

  Args:
    data_array: np.int64 1-d numpy array of shape (size,)
    batch_size: int;
    seq_len: int; length of the rnn layer
    batch_first: boolean; the returned numpy arrays will be of shape
      (batch_size*steps, seq_len) if True and
      (seq_len*steps, batch_size) if False

  Returns:
    named tuple of features and labels, features and labels are np.int64 2-d
      numpy arrays of shape (batch_size*steps, seq_len) if batch_first is True
      and (seq_len*steps, batch_size) if batch_first is False
    steps: int; number of mini batches in an epoch

  Raises:
    ValueError: If input data_array is not 1-d

  """
    if len(data_array.shape) != 1:
        raise ValueError(
            "Expected 1-d data array, "
            "instead data array shape is {} ".format(data_array.shape)
        )

    def fold(used_array):
        shaped_array = np.reshape(used_array, (batch_size, seq_len * steps), order="C")

        if batch_first:
            return np.concatenate(np.split(shaped_array, steps, axis=1), axis=0)
        else:
            return np.transpose(shaped_array)

    steps = (data_array.shape[0] - 1) // (batch_size * seq_len)
    used = batch_size * seq_len * steps

    features = fold(data_array[:used])
    labels = fold(data_array[1 : used + 1])

    Data = collections.namedtuple("Data", ["features", "labels"])
    return Data(features=features, labels=labels), steps


def load_data(debug=False):

    batch_size = 20
    seq_len = 20

    ptrain = os.path.join(HERE, "data", "ptb.train.txt")
    pvalid = os.path.join(HERE, "data", "ptb.valid.txt")
    # ptest = os.path.join(HERE, "data", "ptb.test.txt")
    with open(ptrain, "r") as f1, open(pvalid, "r") as f2:  # , open(ptest, "r") as f3:
        seq_train = f1.read().replace("\n", "<eos>").split(" ")
        seq_valid = f2.read().replace("\n", "<eos>").split(" ")
        # seq_test = f3.read().replace("\n", "<eos>").split(" ")

    seq_train = list(filter(None, seq_train))
    seq_valid = list(filter(None, seq_valid))
    # seq_test = list(filter(None, seq_test))

    size_train = len(seq_train)
    size_valid = len(seq_valid)
    # size_test = len(seq_test)
    logging.info("size_train {}, size_valid {}".format(size_train, size_valid))

    vocab_train = set(seq_train)
    vocab_valid = set(seq_valid)
    # vocab_test = set(seq_test)

    assert vocab_valid.issubset(vocab_train)
    # assert vocab_test.issubset(vocab_train)
    logging.info(
        "vocab_train {}, vocab_valid {}".format(len(vocab_train), len(vocab_valid))
    )

    vocab_train = sorted(vocab_train)  # must have deterministic ordering, so word2id
    # dictionary is reproducible across invocations
    word2id = {w: i for i, w in enumerate(vocab_train)}
    id2word = {i: w for i, w in enumerate(vocab_train)}

    # *******************************************************************************
    # -- data: np.int64 1-d numpy arrays -> np.int64 2-d numpy arrays of shape ----*
    # -- (seq_len*steps, batch_size) ----------------------------------------------*
    #                                                                              *
    # *******************************************************************************
    # Note tf.contrib.cudnn_rnn.CudnnLSTM requires input tensor to be of shape
    # (seq_len,batch_size,embedding_dim), where as tf.keras.layers.CuDNNLSTM
    # requires input tensor to be of shape (batch_size,seq_len,embedding_dim)
    ids_train = np.array([word2id[word] for word in seq_train], copy=False, order="C")
    ids_valid = np.array([word2id[word] for word in seq_valid], copy=False, order="C")
    # ids_test = np.array([word2id[word] for word in seq_test], copy=False, order="C")

    data_train, steps_train = features_labels(
        ids_train, batch_size, seq_len, batch_first=False
    )
    data_valid, steps_valid = features_labels(
        ids_valid, batch_size, seq_len, batch_first=False
    )
    # data_test, steps_test = features_labels(
    #     ids_test, batch_size, seq_len, batch_first=False
    # )

    X_train, y_train = data_train[0], data_train[1]
    X_valid, y_valid = data_valid[0], data_valid[1]

    if debug:
        X_train, y_train = X_train[:100], y_train[:100]
        X_valid, y_valid = X_valid[:100], y_valid[:100]

    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid)


if __name__ == "__main__":
    load_data(debug=True)
