import json
import os
from datetime import datetime
import inspect

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from deephyper.nas.run.util import create_dir

width = 8
height = width/1.618
fontsize = 18
matplotlib.rcParams.update({
    'font.size': fontsize,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 72,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})

HERE = os.path.dirname(os.path.abspath(__file__))
FILE_EXTENSION = "pdf"

def yaml_load(path):
    with open(path, "r") as f:
        yaml_data = yaml.load(f, Loader=Loader)
    return yaml_data

def load_json(f):
    with open(f, "r") as f:
        js_data = json.load(f)
    return js_data

def load_data_from_exp(path_exp):
    path_log = os.path.join(path_exp, "infos", "deephyper.log")
    path_history = os.path.join(path_exp, "infos", "history")

    # read T0 (initial time)
    with open(path_log, "r") as f:
        date_time_str = f.readline().split("|")[0]
        t0 = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

    def load_infos(path):
        fname = path.split("/")[-1][:-5]
        date_time_str, hash_arch_seq = fname.split("oo")
        date_time_obj = datetime.strptime(date_time_str, '%d-%b-%Y_%H-%M-%S')
        arch_seq = [int(el) for el in hash_arch_seq.split("_")]
        history = load_json(path)

        data = dict(
            time=date_time_obj,
            arch_seq=arch_seq,
            **history
        )

        return data

    history_data = [load_infos(os.path.join(path_history, f)) for f in os.listdir(path_history)]
    history_data = sorted(history_data, key=lambda d: d["time"].timestamp())

    data = {
        "t0": t0,
        "history": history_data
    }

    return data

def plot_objective(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    t0 = data["t0"]

    x_times = [d["time"].timestamp()-t0.timestamp() for d in data["history"]]
    y_val_r2 = [d["val_r2"][-1] for d in data["history"]]
    y_val_r2_max = [max(d["val_r2"]) for d in data["history"]]

    # print("Max R2: ", max(y_val_r2_max))

    def only_max(values):
        res = [values[0]]
        for value in values[1:]:
            res.append(max(res[-1], value))
        return res

    plt.figure()
    plt.scatter(x_times, y_val_r2, label="last")
    plt.scatter(x_times, y_val_r2_max, label="max")
    plt.plot(x_times, only_max(y_val_r2))

    plt.ylabel("$R^2$")
    plt.xlabel("Time (Sec.)")
    plt.ylim(-1,1)
    plt.xlim(0,3600*3)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_training_time(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    t0 = data["t0"]

    x_times = [d["time"].timestamp()-t0.timestamp() for d in data["history"]]
    y_training_time = [d["training_time"] for d in data["history"]]

    plt.figure()
    plt.scatter(x_times, y_training_time)
    plt.xlim(0,3600*3)
    plt.ylabel("Training Time (Sec.)")
    plt.xlabel("Time (Sec.)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)

def generate_plot_from_exp(dataset, experiment, output_path):
    print(f"Generating plot for {dataset}.{experiment}")

    module_path = os.path.join(os.path.dirname(HERE), "nas_big_data", dataset)
    experiment_path = os.path.join(module_path, "exp", experiment)
    print(f"Experimental Data Located at: {experiment_path}")

    data = load_data_from_exp(experiment_path)

    # List of plots
    plot_objective(data, output_path)
    plot_training_time(data, output_path)


def plot_objective_multi(experiments, output_path, baseline_data=None):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    def only_max(values):
        res = [values[0]]
        for value in values[1:]:
            res.append(max(res[-1], value))
        return res

    xmin, xmax = 0, 3600*3
    plt.figure()

    if baseline_data:
        plt.hlines(max(baseline_data["val_r2"]), xmin, xmax, colors="black", linestyles="--", label="baseline")

    for exp,data in experiments.items():
        t0 = data["t0"]

        x_times = [d["time"].timestamp()-t0.timestamp() for d in data["history"]]
        y_val_r2 = [d["val_r2"][-1] for d in data["history"]]
        y_val_r2_max = [max(d["val_r2"]) for d in data["history"]]

        plt.plot(x_times, only_max(y_val_r2), label=exp)

    plt.ylabel("$R^2$")
    plt.xlabel("Time (Sec.)")
    plt.ylim(0.85,0.95)
    plt.xlim(xmin,xmax)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_scatter_objective_multi(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    for exp,data in experiments.items():
        t0 = data["t0"]

        x_times = [d["time"].timestamp()-t0.timestamp() for d in data["history"]]
        y_val_r2 = [d["val_r2"][-1] for d in data["history"]]

        plt.scatter(x_times, y_val_r2, alpha=1.0, s=2, label=" ".join(exp.split("_")[1:]))

    plt.ylabel("$R^2$")
    plt.xlabel("Time (Sec.)")
    plt.ylim(0.5,1)
    plt.xlim(0,3600*3)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_best_objective(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exps = []
    best_obj = []
    for exp,data in experiments.items():

        y_val_r2 = [d["val_r2"][-1] for d in data["history"]]
        best_obj.append(max(y_val_r2))
        exps.append("\n".join(exp.split("_")[1:]))

    plt.bar(exps, best_obj)
    plt.ylabel("$R^2$")
    plt.ylim(0.9,0.95)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_number_of_evaluations(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exps = []
    number_evaluations = []
    for exp, data in experiments.items():

        number_evaluations.append(len(data["history"]))
        exps.append("\n".join(exp.split("_")[1:]))

    plt.bar(exps, number_evaluations)
    plt.ylabel("#Evaluations")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_usage_training_time(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    max_time_usage = 8 * 8 * 3600 * 3

    plt.figure()

    exps = []
    training_times = []
    for exp,data in experiments.items():
        t0 = data["t0"]
        ngpus = int(exp.split("gpu")[0].split("_")[-1])
        cum_training_time = sum([d["training_time"]*ngpus for d in data["history"]])
        exps.append("\n".join(exp.split("_")[1:]))
        training_times.append(cum_training_time/max_time_usage*100)

    plt.bar(exps, training_times)
    plt.ylim(0,100)
    plt.ylabel("Time (Sec.)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_best_networks(data, baseline_data, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    plt.plot(list(range(len(baseline_data["val_r2"]))), baseline_data["val_r2"], "k--", label="baseline")

    for exp_name, exp_data in data.items():

        run_data = exp_data[0]
        plt.plot(list(range(len(run_data["val_r2"]))), run_data["val_r2"], label=exp_name)

    plt.ylabel("val_r2")
    plt.xlabel("epochs")
    plt.ylim(0.6,1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)


def generate_plot_from_dataset(dataset, experiments, output_path):

    module_path = os.path.join(os.path.dirname(HERE), "nas_big_data", dataset)

    # check baseline data:
    baseline_data = None
    if "baseline" in experiments:
        baseline_file = experiments["baseline"]
        baseline_path = os.path.join(module_path, "baseline", baseline_file)
        baseline_data = load_json(baseline_path)

    bests_data = None
    bests_output_path = None
    if "bests" in experiments:
        bests_output_path = os.path.join(output_path, "bests")
        create_dir(bests_output_path)

        bests_data = {}
        for best_name in experiments["bests"]:
            hists_path = os.path.join(module_path, "best", best_name, "history")
            hists = [name for name in os.listdir(hists_path) if "json" in name]
            bests_data[best_name] = [load_json(os.path.join(hists_path, h)) for h in hists]

    # load the data for each experiment
    experiments = {exp:None for exp in experiments["experiments"]}
    for exp in experiments:
        experiment_path = os.path.join(module_path, "exp", exp)
        data = load_data_from_exp(experiment_path)
        experiments[exp] = data

    plot_objective_multi(experiments, output_path, baseline_data)
    plot_best_objective(experiments, output_path)
    plot_number_of_evaluations(experiments, output_path)
    plot_scatter_objective_multi(experiments, output_path)
    plot_usage_training_time(experiments, output_path)


    # plot best
    if bests_data is not None:
        plot_best_networks(bests_data, baseline_data, bests_output_path)


def main():

    output_path = os.path.join(HERE, "outputs")
    create_dir(output_path)

    experiments_path = os.path.join(HERE, "experiments.yaml")
    experiments = yaml_load(experiments_path)

    for dataset in experiments:

        dataset_path = os.path.join(output_path, dataset)
        create_dir(dataset_path)

        for experiment in experiments[dataset]["experiments"]:

            experiment_path = os.path.join(dataset_path, experiment)
            create_dir(experiment_path)

            generate_plot_from_exp(dataset, experiment, output_path=experiment_path)

        # comparative plots between experiments of the same dataset
        generate_plot_from_dataset(dataset, experiments[dataset], output_path=dataset_path)

if __name__ == "__main__":
    main()
