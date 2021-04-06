import os
import gzip
import numpy as np
import h5py

from nas_big_data.data_utils import cache_load_data_h5

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data_h5(split="train"):

    h5f_path = os.path.join(HERE, "training_attn.h5")
    h5f = h5py.File(h5f_path, "r")

    if split == "train":
        X, y = h5f["X_train"][:], h5f["Y_train"][:]
    elif split == "valid":
        X, y = h5f["X_val"][:], h5f["Y_val"][:]
    elif split == "test":
        X, y = h5f["X_test"][:], h5f["Y_test"][:]

    h5f.close()

    y = np.argmax(y, axis=1)

    return X, y


def load_data_test():

    X_train, y_train = load_data_h5("train")
    X_valid, y_valid = load_data_h5("valid")
    X_train = np.concatenate([X_train, X_valid], axis=0)
    y_train = np.concatenate([y_train, y_valid], axis=0)
    X_test, y_test = load_data_h5("test")

    return (X_train, y_train), (X_test, y_test)


def load_data():

    X_train, y_train = load_data_h5("train")
    X_valid, y_valid = load_data_h5("valid")

    return (X_train, y_train), (X_valid, y_valid)


@cache_load_data_h5("/dev/shm/attn.h5")
def load_data_cache():
    return load_data()


def test_load_data_cache():
    from time import time

    t1 = time()
    load_data()
    t2 = time()
    dur = t2 - t1
    print("Normal loading: ", dur)  # -> ~ 35 sec

    t1 = time()
    load_data_cache()
    t2 = time()
    dur = t2 - t1
    print("Cache call 1 loading: ", dur)  # -> 45 sec

    t1 = time()
    load_data_cache()
    t2 = time()
    dur = t2 - t1
    print("Cache call 2 loading: ", dur)  # -> 2 sec


if __name__ == "__main__":
    # load_data()
    test_load_data_cache()
