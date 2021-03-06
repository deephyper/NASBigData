"""
OpenML results: https://www.openml.org/t/218
"""
from sklearn import preprocessing
import numpy as np

from deephyper.benchmark.datasets import dionis
from deephyper.benchmark.datasets.util import cache_load_data
from deephyper.nas.preprocessing import minmaxstdscaler


@cache_load_data("/dev/shm/dionis.npz")
def load_data_cache(use_test=False):
    # Random state
    random_state = np.random.RandomState(seed=42)

    if use_test:
        print("!!! USING TEST DATA !!!")
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dionis.load_data(
            random_state=random_state, test_size=0.33, valid_size=0.33 * (1 - 0.33)
        )
        X_train = np.concatenate([X_train, X_valid])
        y_train = np.concatenate([y_train, y_valid])
        X_valid, y_valid = X_test, y_test
    else:
        (X_train, y_train), (X_valid, y_valid), _ = dionis.load_data(
            random_state=random_state, test_size=0.33, valid_size=0.33 * (1 - 0.33)
        )

    prepro_output = preprocessing.OneHotEncoder()
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_train = prepro_output.fit_transform(y_train).toarray()
    y_valid = prepro_output.transform(y_valid).toarray()

    prepro_input = minmaxstdscaler()
    X_train = prepro_input.fit_transform(X_train)
    X_valid = prepro_input.transform(X_valid)

    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid)


def load_data(use_test=False):
    return load_data_cache(use_test=use_test)


def test_baseline():
    """Test data with RandomForest

    accuracy_score on Train:  1.0
    accuracy_score on Test:  0.9408463203126216
    balanced_acc on Train:  1.0
    balanced_acc on Test:  0.877023562682862
    """
    from sklearn.ensemble import RandomForestClassifier
    from deephyper.baseline import BaseClassifierPipeline
    from sklearn.utils import class_weight
    from sklearn import metrics

    def load_data():
        train, valid, _ = dionis.load_data(random_state=42)
        return train, valid

    train, valid = load_data()
    prop_train = np.bincount(train[1]) / len(train[1])
    prop_valid = np.bincount(valid[1]) / len(valid[1])

    print("classes: ", np.unique(train[1]))
    print("prop_train: ", prop_train)
    print("prop_valid: ", prop_valid)

    baseline_classifier = BaseClassifierPipeline(
        RandomForestClassifier(n_jobs=6), load_data
    )
    baseline_classifier.run()

    def balanced_acc(y_true, y_pred):
        cw = class_weight.compute_class_weight("balanced", np.unique(y_true), y_true)
        sw = np.array([cw[class_ - 1] for class_ in y_true])
        bacc = metrics.accuracy_score(y_true, y_pred, sample_weight=sw)
        return bacc

    baseline_classifier.evaluate(balanced_acc)


if __name__ == "__main__":
    load_data(use_test=True)
    load_data(use_test=False)
    # test_baseline()
