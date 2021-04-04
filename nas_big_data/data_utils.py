import os
import h5py
import numpy as np


def convert_to_dataframe(X, y):
    """Convert a X(inputs), y(outputs) dataset to a dataframe format.

    Args:
        X (np.array): inputs.
        y (np.array): labels.

    Returns:
        DataFrame: a Pandas Dataframe with inputs named as "xi" and labels named as "label".
    """
    import numpy as np
    import pandas as pd

    data = np.concatenate([X, np.argmax(y, axis=1).reshape(-1, 1)], axis=1)
    df = pd.DataFrame(
        data=data, columns=[f"x{i}" for i in range(np.shape(data)[-1] - 1)] + ["label"]
    )

    return df


def cache_load_data(cache_loc):
    """Decorator of load_data function to dache numpy arrays return by the function. The load_data function should return a tuple of the form: ``(X_train, y_train), (X_valid, y_valid)``.

    Args:
        cache_loc (str): path where the data will be cached.
    """

    def _cache(data_loader):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_loc):
                print("Reading data from cache")
                with open(cache_loc, "rb") as fp:
                    data = {
                        k: arr for k, arr in np.load(fp, allow_pickle=True).item().items()
                    }
                return (
                    (data["X_train"], data["y_train"]),
                    (data["X_valid"], data["y_valid"]),
                )

            else:
                (X_train, y_train), (X_valid, y_valid) = data_loader(*args, **kwargs)
                if os.path.exists(os.path.dirname(cache_loc)):
                    print("Data not cached; invoking user data loader")
                    data = {
                        "X_train": X_train,
                        "y_train": y_train,
                        "X_valid": X_valid,
                        "y_valid": y_valid,
                    }
                    with open(cache_loc, "wb") as fp:
                        np.save(fp, data)
                else:
                    print(
                        "Data cannot be cached because the path does not exist. Returning data anyway."
                    )
                return (X_train, y_train), (X_valid, y_valid)

        return wrapper

    return _cache


def cache_load_data_h5(cache_loc):
    """Decorator of load_data function to dache numpy arrays return by the function. The load_data function should return a tuple of the form: ``(X_train, y_train), (X_valid, y_valid)``.

    Args:
        cache_loc (str): path where the data will be cached.
    """

    def _cache(data_loader):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_loc):
                print("Reading data from cache")
                h5f = h5py.File(cache_loc, "r")
                X_train = h5f["X_train"][:]
                y_train = h5f["y_train"][:]
                X_valid = h5f["X_valid"][:]
                y_valid = h5f["y_valid"][:]
                return (
                    (X_train, y_train),
                    (X_valid, y_valid),
                )

            else:
                print("Data not cached; invoking user data loader.")
                (X_train, y_train), (X_valid, y_valid) = data_loader(*args, **kwargs)
                if os.path.exists(os.path.dirname(cache_loc)):
                    print("Caching Data.")
                    h5f = h5py.File("training_attn.h5", "w")
                    h5f.create_dataset("X_train", data=X_train)
                    h5f.create_dataset("y_train", data=y_train)
                    h5f.create_dataset("X_valid", data=X_valid)
                    h5f.create_dataset("y_valid", data=y_valid)
                    h5f.close()
                else:
                    print(
                        "Data cannot be cached because the path does not exist. Returning data anyway."
                    )
                return (X_train, y_train), (X_valid, y_valid)

        return wrapper

    return _cache