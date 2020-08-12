import argparse
import json
import os
import pathlib
import socket
import time

from autogluon import TabularPrediction as task
from nas_big_data.covertype.load_data import load_data
from nas_big_data.data_utils import convert_to_dataframe

here = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(here, "outputs")


def hostnames_to_ips(hostnames: str) -> list:
    """3826-3827,3830,3832-3833,3836,3838-3839

    Args:
        hostnames (str): "3826-3827,3830,3832-3833,3836,3838-3839"

    Returns:
        list: generator of ip addresses.
    """
    hostnames = hostnames.split(",")

    def to_nid(hn):
        hn = int(hn) if type(hn) is str else hn
        return f"nid{hn:05d}"

    def addresses_generator(hostnames: list) -> str:
        for hn in hostnames:
            if "-" in hn:
                start, end = hn.split("-")
                for hn_ in range(int(start), int(end) + 1):
                    yield socket.gethostbyname(to_nid(hn_))
            else:
                yield socket.gethostbyname(to_nid(hn))

    return addresses_generator(hostnames)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--walltime", type=int, default=30 * 60, help="Walltime to fit AutoGluon"
)
parser.add_argument(
    "--evaluate", const=True, nargs="?", default=False, help="Evaluate model."
)

parser.add_argument(
    "--no-knn", const=True, nargs="?", default=False, help="Deactivate KNN."
)

args = parser.parse_args()

if not args.evaluate:

    # Retriving COBALT infos
    hostnames = os.environ.get("COBALT_PARTNAME", "")
    jobsize = int(os.environ.get("COBALT_JOBSIZE", 0))

    # Building list of ip addresses
    ips = list(hostnames_to_ips(hostnames))

    assert len(ips) == jobsize and jobsize > 0, f"Ips is: {ips}, Jobsize is: {jobsize}"

    ips = ips[1:]

    if args.no_knn <= 120:
        excluded_model_types = ["KNN"]
    else:
        excluded_model_types = []

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_valid, y_valid) = load_data(use_test=False)

    df_train = convert_to_dataframe(X_train, y_train)
    df_valid = convert_to_dataframe(X_valid, y_valid)

    # hyperparameters
    nunits = list(range(16, 97, 16))
    nn_options = {  # specifies non-default hyperparameter values for neural network models
        "num_epochs": 100,  # number of training epochs (controls training time of NN models)
        "learning_rate": ag.space.Real(
            0.001, 0.1, default=0.01, log=True
        ),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
        "activation": ag.space.Categorical(
            None, swish, "relu", "tanh", "sigmoid"
        ),  # activation function used in NN (categorical hyperparameter, default = first entry)
        "layers": ag.space.Categorical(*(nunits for _ in range(10))),
        # Each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
        "dropout_prob": 0.0,
    }
    hyperparameters = {"NN": nn_options}

    predictor = task.fit(
        train_data=task.Dataset(df=df_train),
        # tuning_data=task.Dataset(df=df_valid),
        label="label",
        output_directory=output_dir,
        time_limits=args.walltime,
        hyperparameter_tune=True,
        hyperparameters=hyperparameters,
        auto_stack=True,
        excluded_model_types=excluded_model_types,
        dist_ip_addrs=ips,
    )
else:
    _, (X_test, y_test) = load_data(use_test=True)

    print("Convert arrays to DataFrame...")
    df_test = convert_to_dataframe(X_test, y_test)

    print("Loading models...")
    predictor = task.load(output_dir, verbosity=4)

    print("Predicting...")
    t1 = time.time()
    y_pred = predictor.predict(task.Dataset(df=df_test))
    t2 = time.time()

    y_test = df_test.label

    print("Evaluation predictions...")
    results = predictor.evaluate_predictions(
        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
    )
    print(results)

    test_scores_path = os.path.join(here, "test_scores.json")
    data_json = {"timing_predict": t2 - t1}
    with open("timing_predict.json", "w") as fp:
        json.dump(data_json, fp)
    with open(test_scores_path, "w") as fp:
        json.dump(results, fp)
