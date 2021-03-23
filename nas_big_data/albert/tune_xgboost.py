import json
import os

"""
deephyper hps ambs --problem nas_big_data.albert.tune_xgboost.Problem --run nas_big_data.albert.tune_xgboost.run --max-evals 30 --evaluator subprocess --n-jobs 3
"""
import pandas as pd
from deephyper.problem import HpProblem
from nas_big_data import RANDOM_STATE
from nas_big_data.albert.load_data import load_data
from xgboost import XGBClassifier

Problem = HpProblem()

Problem.add_hyperparameter((10, 300), "n_estimators")
Problem.add_hyperparameter((0.0001, 0.1, "log-uniform"), "learning_rate")
Problem.add_hyperparameter((1, 50), "max_depth")

# We define a starting point with the defaul hyperparameters from sklearn-learn
# that we consider good in average.
Problem.add_starting_point(n_estimators=100, learning_rate=0.001, max_depth=50)


def test_best():
    fcsv = "results_tune_xgboost.csv"
    csv_path = os.path.join(os.path.dirname(__file__), fcsv)
    df = pd.read_csv(csv_path)
    config = df.iloc[df.objective.argmax()][:-2].to_dict()

    config["max_depth"] = int(config["max_depth"])
    config["n_estimators"] = int(config["n_estimators"])
    config["learning_rate"] = float(config["learning_rate"])
    print(config)

    (X_train, y_train), (X_test, y_test) = load_data(use_test=True, out_ohe=False)

    model_kwargs = dict(**config)
    model = XGBClassifier(random_state=RANDOM_STATE, **model_kwargs)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

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
    (X_train, y_train), (X_test, y_test) = load_data(use_test=False, out_ohe=False)

    model_kwargs = dict(n_jobs=6, random_state=RANDOM_STATE, **config)
    model = XGBClassifier(**model_kwargs)

    model.fit(X_train, y_train)

    valid_acc = model.score(X_test, y_test)

    return valid_acc


if __name__ == "__main__":
    print(Problem)
    test_best()
