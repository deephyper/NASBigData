import numpy as np
import tensorflow as tf
import torchvision.transforms as transforms
from sklearn import model_selection, preprocessing

from deephyper.benchmark.datasets.util import cache_load_data


@cache_load_data("/dev/shm/cifar10.npz")
def load_data_cache(verbose: bool = True):
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


def load_data():
    return load_data_cache()


if __name__ == "__main__":
    load_data()
