"""
deephyper hps ambs --problem nas_big_data.albert.tune_random_forest.Problem --run nas_big_data.albert.tune_random_forest.run --max-evals 30 --evaluator subprocess --n-jobs 3
"""

import json
import os

import pandas as pd
from deephyper.problem import HpProblem
from nas_big_data import RANDOM_STATE
from nas_big_data.albert.load_data import load_data
from sklearn.ensemble import RandomForestClassifier

Problem = HpProblem()

Problem.add_hyperparameter((10, 300), "n_estimators")
Problem.add_hyperparameter(["gini", "entropy"], "criterion")
Problem.add_hyperparameter((1, 50), "max_depth")
Problem.add_hyperparameter((2, 10), "min_samples_split")

# We define a starting point with the defaul hyperparameters from sklearn-learn
# that we consider good in average.
Problem.add_starting_point(
    n_estimators=100, criterion="gini", max_depth=50, min_samples_split=2
)


def test_best():
    """Test data with RandomForest

    """
    fcsv = "results_tune_random_forest.csv"
    csv_path = os.path.join(os.path.dirname(__file__), fcsv)
    df = pd.read_csv(csv_path)
    config = df.iloc[df.objective.argmax()][:-2].to_dict()

    (X_train, y_train), (X_test, y_test) = load_data(use_test=True, out_ohe=False)

    model_kwargs = dict(n_jobs=6, **config)
    model = RandomForestClassifier(random_state=RANDOM_STATE, **model_kwargs)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    model_kwargs["max_depth"] = int(model_kwargs["max_depth"])
    model_kwargs["min_samples_split"] = int(model_kwargs["min_samples_split"])
    model_kwargs["n_estimators"] = int(model_kwargs["n_estimators"])

    print(type(model).__name__)
    scores = dict(
        model=type(model).__name__,
        model_args=model_kwargs,
        train_acc=train_acc,
        test_acc=test_acc,
    )

    # save scores to disk
    fname = os.path.basename(__file__)[:-3]
    fpath = os.path.join(os.path.dirname(__file__), fname + ".json")
    with open(fpath, "w") as fp:
        json.dump(scores, fp)


def run(config):
    """Test data with RandomForest

    accuracy_score on Train:  1.0
    accuracy_score on Test:  0.6638969849425279
    """

    (X_train, y_train), (X_test, y_test) = load_data(use_test=False, out_ohe=False)

    model_kwargs = dict(n_jobs=6, random_state=RANDOM_STATE, **config)
    model = RandomForestClassifier(**model_kwargs)

    model.fit(X_train, y_train)

    valid_acc = model.score(X_test, y_test)

    return valid_acc


if __name__ == "__main__":
    # print(Problem)
    test_best()
