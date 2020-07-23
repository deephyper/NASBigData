import os

import numpy as np

from deephyper.benchmark.datasets.util import cache_load_data

HERE = os.path.dirname(os.path.abspath(__file__))


@cache_load_data("/dev/shm/cifar10.npz")
def load_data_cache_v1(verbose: bool = True):
    import torchvision.transforms as transforms
    import tensorflow as tf
    from sklearn import model_selection, preprocessing

    random_state = np.random.RandomState(seed=42)

    (X_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X_train, y_train, test_size=0.33, shuffle=True, random_state=random_state
    )

    prepro_output = preprocessing.OneHotEncoder()
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_train = prepro_output.fit_transform(y_train).toarray()
    y_valid = prepro_output.transform(y_valid).toarray()

    # from: https://github.com/antoyang/NAS-Benchmark/blob/master/DARTS/preproc.py
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf_train = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    transf_val = []
    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_transform = transforms.Compose(transf_train + normalize)
    valid_transform = transforms.Compose(transf_val + normalize)

    X_train = np.array(
        [np.transpose(train_transform(x).numpy(), (1, 2, 0)) for x in X_train]
    )
    X_valid = np.array(
        [np.transpose(valid_transform(x).numpy(), (1, 2, 0)) for x in X_valid]
    )

    if verbose:
        print(f"X_train shape: {np.shape(X_train)}")
        print(f"y_train shape: {np.shape(y_train)}")
        print(f"X_valid shape: {np.shape(X_valid)}")
        print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid)


@cache_load_data("/dev/shm/cifar10.npz")
def load_data_cache_v2(verbose: bool = True):
    with open(os.path.join(HERE, "cifar10.npz"), "rb") as fp:
        data = {k: arr for k, arr in np.load(fp).items()}
    print(f"X_train shape: {np.shape(data['X_train'])}")
    print(f"y_train shape: {np.shape(data['y_train'])}")
    print(f"X_valid shape: {np.shape(data['X_valid'])}")
    print(f"y_valid shape: {np.shape(data['y_valid'])}")
    return (data["X_train"], data["y_train"]), (data["X_valid"], data["y_valid"])


def load_data():
    return load_data_cache_v2()


if __name__ == "__main__":
    load_data()
