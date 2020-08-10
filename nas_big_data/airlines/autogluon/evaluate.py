import os
import pathlib
import argparse
import json

from autogluon import TabularPrediction as task
from nas_big_data.airlines.load_data import load_data
from nas_big_data.data_utils import convert_to_dataframe

here = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(here, "outputs")

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--walltime", type=int, default=30 * 60, help="Walltime to fit AutoGluon"
)
parser.add_argument(
    "--evaluate", const=True, nargs="?", default=False, help="Evaluate model."
)

args = parser.parse_args()

if not args.evaluate:
    if args.walltime <= 120:
        excluded_model_types = ["KNN"]
    else:
        excluded_model_types = []

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_valid, y_valid) = load_data(use_test=False)

    df_train = convert_to_dataframe(X_train, y_train)
    df_valid = convert_to_dataframe(X_valid, y_valid)

    predictor = task.fit(
        train_data=task.Dataset(df=df_train),
        tuning_data=task.Dataset(df=df_valid),
        label="label",
        output_directory=output_dir,
        time_limits=args.walltime,
        hyperparameter_tune=True,
        auto_stack=True,
        excluded_model_types=excluded_model_types,
    )
else:
    _, (X_test, y_test) = load_data(use_test=True)

    print("Convert arrays to DataFrame...")
    df_test = convert_to_dataframe(X_test, y_test)

    print("Loading models...")
    predictor = task.load(output_dir, verbosity=4)

    print("Predicting...")
    y_pred = predictor.predict(task.Dataset(df=df_test))

    y_test = df_test.label

    print("Evaluation predictions...")
    results = predictor.evaluate_predictions(
        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
    )
    print(results)

    test_scores_path = os.path.join(here, "test_scores.json")
    with open(test_scores_path, "w") as fp:
        json.dump(results, fp)
