"""
OpenML results: https://www.openml.org/t/218
"""
from types import new_class
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from deephyper.benchmark.datasets import albert
from deephyper.benchmark.datasets.util import cache_load_data
from deephyper.nas.preprocessing import minmaxstdscaler

from nas_big_data import RANDOM_STATE


# @cache_load_data("/dev/shm/albert.npz")
def load_data_cache(use_test=False, out_ohe=True):

    test_size = 0.33
    valid_size = 0.33 * (1 - test_size)
    if use_test:
        print("!!! USING TEST DATA !!!")
        (
            (X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test),
            categorical_indicator,
        ) = albert.load_data(
            random_state=RANDOM_STATE,
            test_size=test_size,
            valid_size=valid_size,
            categoricals_to_integers=True,
        )
        X_train = np.concatenate([X_train, X_valid])
        y_train = np.concatenate([y_train, y_valid])
        X_valid, y_valid = X_test, y_test
    else:
        (
            (X_train, y_train),
            (X_valid, y_valid),
            _,
            categorical_indicator,
        ) = albert.load_data(
            random_state=RANDOM_STATE,
            test_size=test_size,
            valid_size=valid_size,
            categoricals_to_integers=True,
        )

    # Replace missing values with mean value
    # https://scikit-learn.org/stable/modules/impute.html
    print("Replacing missing values")
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train = imp_mean.fit_transform(X_train)
    X_valid = imp_mean.transform(X_valid)

    # Min Max => Std scaler preprocessing for non categorical variables
    for i, (categorical, _) in enumerate(categorical_indicator):
        if not categorical:
            scaler = minmaxstdscaler()
            X_train[:, i : i + 1] = scaler.fit_transform(X_train[:, i : i + 1])
            X_valid[:, i : i + 1] = scaler.transform(X_valid[:, i : i + 1])

    # One Hot Encoding of Outputs
    if out_ohe:
        prepro_output = preprocessing.OneHotEncoder()
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)
        y_train = prepro_output.fit_transform(y_train).toarray()
        y_valid = prepro_output.transform(y_valid).toarray()

    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid), categorical_indicator


def load_data(use_test=False, out_ohe=True):
    return load_data_cache(use_test=use_test, out_ohe=out_ohe)


if __name__ == "__main__":
    # load_data(use_test=True)
    load_data(use_test=False)
