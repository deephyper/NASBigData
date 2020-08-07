import os
import pathlib
import argparse

from autogluon import TabularPrediction as task
from nas_big_data.covertype.load_data import load_data
from nas_big_data.data_utils import convert_to_dataframe

here = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(here, "outputs")

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--walltime", type=int, default=30 * 60, help="Walltime to fit AutoGluon"
)

args = parser.parse_args()

# Create output directory
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

(X_train, y_train), (X_test, y_test) = load_data(use_test=True)

df_train = convert_to_dataframe(X_train, y_train)

predictor = task.fit(
    train_data=task.Dataset(df=df_train),
    label="label",
    output_directory=output_dir,
    time_limits=args.walltime,
)

df_test = convert_to_dataframe(X_test, y_test)

y_pred = predictor.predict(task.Dataset(df=df_test))

y_test = df_test.label

perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
)
print(perf)
