import inspect
import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.spatial import distance
import pandas as pd
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from deephyper.nas.run.util import create_dir

METRIC = None
METRIC_TO_LABEL = {
    "val_r2": "Validation $R^2$",
    "val_auc": "Validation AUC",
    "val_acc": "Validation accuracy",
    "val_aucpr": "Validation AUC PR",
    "val_auroc": "Validation AU ROC",
}
METRIC_LIMITS = []
EXPNAME_TO_LABEL = {}
EXPNAME_BEST_TO_LABEL = {}
TMIN, TMAX = 0, 3600 * 3

width = 8
height = width / 1.618
fontsize = 25
legend_font_size = 18
# fontsize = 18
# legend_font_size = 12
matplotlib.rcParams.update(
    {
        "font.size": fontsize,
        "figure.figsize": (width, height),
        "figure.facecolor": "white",
        "savefig.dpi": 72,
        "figure.subplot.bottom": 0.125,
        "figure.edgecolor": "white",
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "lines.linewidth": 3,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
FILE_EXTENSION = "pdf"


@ticker.FuncFormatter
def hour_major_formatter(x, pos):
    x = float(f"{x/3600:.1f}")
    if x % 1 == 0:
        x = str(int(x))
    else:
        x = f"{x:.1f}"
    return x


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
    path_csv = os.path.join(path_exp, "infos", "results.csv")

    # read T0 (initial time)
    with open(path_log, "r") as f:
        date_time_str = f.readline().split("|")[0]
        t0 = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

    def load_infos(path):
        fname = path.split("/")[-1][:-5]
        date_time_str, hash_arch_seq = fname.split("oo")
        date_time_obj = datetime.strptime(date_time_str, "%d-%b-%Y_%H-%M-%S")
        arch_seq = [int(el) for el in hash_arch_seq.split("_")]
        history = load_json(path)

        data = dict(time=date_time_obj, arch_seq=arch_seq, **history)

        return data

    history_data = [
        load_infos(os.path.join(path_history, f)) for f in os.listdir(path_history)
    ]
    history_data = sorted(history_data, key=lambda d: d["time"].timestamp())

    data = {"t0": t0, "history": history_data, "results": pd.read_csv(path_csv)}

    return data


def plot_objective(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    t0 = data["t0"]

    x_times = [d["time"].timestamp() - t0.timestamp() for d in data["history"]]
    y_val_r2 = [d[METRIC][-1] for d in data["history"]]
    y_val_r2_max = [max(d[METRIC]) for d in data["history"]]

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

    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.xlabel("Time (Sec.)")
    plt.ylim(*METRIC_LIMITS)
    plt.xlim(0, 3600 * 3)
    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_time(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    t0 = data["t0"]

    x_times = [d["time"].timestamp() - t0.timestamp() for d in data["history"]]
    y_training_time = [d["training_time"] for d in data["history"]]

    plt.figure()
    plt.scatter(x_times, y_training_time)
    plt.xlim(0, 3600 * 3)
    plt.ylabel("Training Time (Sec.)")
    plt.xlabel("Time (Sec.)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_diversity_of_arch_seq(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    t0 = data["t0"]

    x_times = np.array([t for t in data["results"].elapsed_sec])/3600
    if "arch_seq" in data["results"].columns:
        archs = np.array([json.loads(arch) for arch in data["results"].arch_seq])
    else:
        archs = data["results"].to_numpy()[:,:-2]

    # PCA
    y = np.array(archs)
    from sklearn import decomposition

    pca = decomposition.PCA(n_components=2)
    y = pca.fit_transform(archs)

    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # print("Cum variance: ", pca.explained_variance_ratio_.cumsum()[1])

    # UMAP
    # import umap

    # T-SNE
    # from sklearn.manifold import TSNE
    # tsne = TSNE()
    # y = tsne.fit_transform(archs)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # plt.sca(ax)

    plt.scatter(y[:,0], y[:,1], c=x_times)
    # ax.scatter(y[:,0], y[:,1], y[:, 2], s=2, alpha=0.8, c=x_times)
    # plt.xlim(0, 3600 * 3)
    # plt.ylabel("Training Time (Sec.)")
    # plt.xlabel("Time (Sec.)")
    # plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_diversity2_of_arch_seq(data, exp_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(exp_path, output_file_name)

    step = 3600/2
    times = np.array([t for t in data["results"].elapsed_sec])#/3600
    if "arch_seq" in data["results"].columns:
        archs = np.array([json.loads(arch) for arch in data["results"].arch_seq])
    else:
        archs = data["results"].to_numpy()[:,:-2]


    x, y = [], []
    for i in range(int(3600*3/step)+1):
        t1 = i*step
        t2 = (i+1)*step
        selection = np.logical_and(t1 <= times, times < t2)
        arch_selec = archs[selection, :]
        # print(f"{int(t1)}-{int(t2)}:", np.shape(archs), np.shape(arch_selec))
        t = (t2+t1)/2
        arch_mean = np.mean(arch_selec, axis=0)
        arch_var = np.sum([distance.cosine(arch_mean, a) for a in arch_selec])
        x.append(t)
        y.append(arch_var)

    plt.figure()

    plt.plot(x, y)
    plt.xlim(0, 3600 * 3)
    plt.ylabel("Diversity")
    plt.xlabel("Time (Sec.)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def output_best_architecture(data, output_path):
    df = data["results"]
    i = df.objective.argmax()
    row = df.iloc[i]
    objective = float(row[-2])
    elapsed_time = float(row[-1])
    hp_names = None

    if "arch_seq" in df.columns:
        arch_seq = row["arch_seq"]
        hp_names = df.columns.tolist()[1:-2]
        row_as_dict = json.loads(row.to_json())
        hp_values = [row_as_dict[name] for name in hp_names]
    else:
        arch_seq = row.tolist()[:-2]

        if all([el % 1 == 0 for el in arch_seq]):
            arch_seq = [int(el) for el in arch_seq]

        arch_seq = str(arch_seq)

    # look for similar architecture in history
    training_time = None
    n_epochs_trained = None
    for d in data["history"]:
        if str(d["arch_seq"]) == arch_seq:
            training_time = d["training_time"] / 60
            n_epochs_trained = len(d["loss"])

    data = {
        "objective": objective,
        "elapsed_time": elapsed_time,
        "arch_seq": arch_seq,
        "training_time": training_time,
        "n_epochs_trained": n_epochs_trained,
    }

    if hp_names is not None:
        data["hyperparameters"] = {
            hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_values)
        }

    output_path = os.path.join(output_path, "best_configuration.yaml")
    with open(output_path, "w") as f:
        yaml.dump(data, f)

    exp_name = output_path.split("/")[-2]
    print(EXPNAME_TO_LABEL[exp_name], " -> ", f"{objective:.3f} {METRIC}")


def generate_plot_from_exp(dataset, experiment, experiment_folder, output_path):
    print(f"Generating plot for {dataset}.{experiment}")

    module_path = os.path.join(os.path.dirname(HERE), "nas_big_data", dataset)
    experiment_path = os.path.join(module_path, experiment_folder, experiment)
    print(f"Experimental Data Located at: {experiment_path}")

    data = load_data_from_exp(experiment_path)

    # List of plots
    plot_objective(data, output_path)
    plot_training_time(data, output_path)
    # plot_diversity_of_arch_seq(data, output_path)
    # plot_diversity2_of_arch_seq(data, output_path)

    # output best architecture and hp
    output_best_architecture(data, output_path)


def plot_objective_multi(experiments, output_path, baseline_data=None):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    def only_max(values):
        res = [values[0]]
        for value in values[1:]:
            res.append(max(res[-1], value))
        return res

    xmin, xmax = TMIN, TMAX
    plt.figure()

    if baseline_data:
        plt.hlines(
            max(baseline_data[METRIC]),
            xmin,
            xmax,
            colors="black",
            linestyles="--",
            label="baseline",
        )

    for exp, data in experiments.items():
        t0 = data["t0"]

        x_times = [d["time"].timestamp() - t0.timestamp() for d in data["history"]]
        y_val_r2 = [d[METRIC][-1] for d in data["history"]]

        x_times, y_val_r2 = list(zip(*sorted(zip(x_times, y_val_r2), key=lambda t: t[0])))

        plt.plot(x_times, only_max(y_val_r2), label=EXPNAME_TO_LABEL[exp], alpha=0.8)

    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.xlabel("Time (Hour)")
    plt.ylim(*METRIC_LIMITS)
    plt.xlim(xmin, xmax)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))
    ax.xaxis.set_major_formatter(hour_major_formatter)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    plt.grid()
    plt.legend(fontsize=legend_font_size, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_diversity2_multi(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    xmin, xmax = TMIN, TMAX
    step = 3600/6

    plt.figure()

    for exp, data in experiments.items():

        times = np.array([t for t in data["results"].elapsed_sec])
        if "arch_seq" in data["results"].columns:
            archs = np.array([json.loads(arch) for arch in data["results"].arch_seq])
        else:
            archs = data["results"].to_numpy()[:,:-2]

        x, y = [], []
        for i in range(int(TMAX/step)+1):
            t1, t2 = i*step, (i+1)*step
            t = (t2+t1)/2
            selection = np.logical_and(t1 <= times, times < t2)
            arch_selec = archs[selection, :]
            if len(arch_selec) == 0:
                break
            arch_mean = np.mean(arch_selec, axis=0)
            arch_var = np.sum([distance.cosine(arch_mean, a) for a in arch_selec])
            # arch_var = np.sum([distance.euclidean(arch_mean, a) for a in arch_selec])
            x.append(t)
            y.append(arch_var)

        plt.plot(x, y, label=EXPNAME_TO_LABEL[exp])

    plt.ylabel("Diversity")
    plt.xlabel("Time (Hour)")
    plt.xlim(xmin, xmax)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800*2))
    ax.xaxis.set_major_formatter(hour_major_formatter)

    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter_objective_multi(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    for exp, data in experiments.items():
        t0 = data["t0"]

        x_times = [d["time"].timestamp() - t0.timestamp() for d in data["history"]]
        y_val_r2 = [d[METRIC][-1] for d in data["history"]]

        plt.scatter(x_times, y_val_r2, alpha=1.0, s=2, label=" ".join(exp.split("_")[1:]))

    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.xlabel("Time (Sec.)")
    plt.ylim(*METRIC_LIMITS)
    plt.xlim(0, 3600 * 3)
    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_objective(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exps = []
    best_obj = []
    for exp, data in experiments.items():

        y_val_r2 = [d[METRIC][-1] for d in data["history"]]
        best_obj.append(max(y_val_r2))
        exps.append("\n".join(exp.split("_")[1:]))

    plt.bar(exps, best_obj)
    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.ylim(*METRIC_LIMITS)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_number_of_evaluations(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exps = []
    number_evaluations = []
    for exp, data in experiments.items():

        number_evaluations.append(len(data["history"]))
        # exps.append("\n".join(exp.split("_")[1:]))
        exps.append(EXPNAME_TO_LABEL[exp])

        print(exps[-1].replace("\n", " "), " -> ", number_evaluations[-1], "evaluations")

    plt.bar(exps, number_evaluations)
    plt.ylabel("#Evaluations")
    plt.xticks(rotation=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_usage_training_time(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    # max_time_usage = 8 * 8 * 3600 * 3
    wall_time = 3600 * 3

    plt.figure()

    exps = []
    training_times = []
    for exp, data in experiments.items():
        ngpus = int(exp.split("gpu")[0].split("_")[-1])
        nnodes = int(exp.split("_")[2])
        max_time_usage = 8 * nnodes * wall_time
        cum_training_time = sum([d["training_time"] * ngpus for d in data["history"]])
        # exps.append("\n".join(exp.split("_")[1:]))
        exps.append(EXPNAME_TO_LABEL[exp])
        training_times.append(cum_training_time / max_time_usage * 100)
        print(f"Usage: {exps[-1]} -> {training_times[-1]}%")

        serie_times = [d["training_time"] for d in data["history"]]
        mean_time = np.mean(serie_times)
        std_time = np.std(serie_times)
        print(exp, " -> ", f"{mean_time/60:.2f} ± {std_time/60:.2f}")

    plt.bar(exps, training_times)
    plt.ylim(0, 100)
    plt.ylabel("Time in Training (%)")
    plt.xticks(rotation=35)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_distribution_training_duration(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    wall_time = 3600 * 3

    plt.figure()

    df = pd.DataFrame(data={
        "training_time": [],
        "exp_name": []
    })

    for exp, data in experiments.items():
        serie_times = [d["training_time"]/60 for d in data["history"]]
        serie_objectives = [d[METRIC][-1] for d in data["history"]]
        # plt.hist(serie_times, bins=20, density=True, label=exp, alpha=0.5)
        newdf = pd.DataFrame(data={
            "training_time": serie_times,
            "exp_name": [EXPNAME_TO_LABEL[exp] for _ in range(len(serie_times))],
            "objective": serie_objectives
        })
        # newdf = newdf.sort_values("objective", ascending=False).iloc[:100]
        # print(newdf.objective[:10])
        df = pd.concat([df, newdf], ignore_index=True)

    # sns.displot(df, x="training_time", hue="exp_name", kind="kde")
    sns.boxplot(x="exp_name", y="training_time", data=df)

    plt.xlabel("")
    plt.ylabel("Training Time (min.)")
    plt.yscale("log")
    plt.grid()
    plt.xticks(rotation=35)
    plt.tight_layout()
    # plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_count_arch_better_than_baseline(experiments, output_path, baseline_data=None):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    if baseline_data is None:
        return

    plt.figure()

    base_r2 = max(baseline_data[METRIC])

    for exp_name, exp_data in experiments.items():

        t0 = exp_data["t0"]
        x = [d["time"].timestamp() - t0.timestamp() for d in exp_data["history"] if d[METRIC][-1] > base_r2]
        y = [1 for d in exp_data["history"] if d[METRIC][-1] > base_r2]
        arch_list = [d["arch_seq"] for d in exp_data["history"] if d[METRIC][-1] > base_r2]

        if len(x) > 0:
            x, y, arch_list = zip(*sorted(zip(x, y, arch_list), key=lambda t: t[0]))
        else:
            x, y = [], []
        x, y = list(x), list(y)

        seen_arch_seq = []
        for i, arch_seq in enumerate(arch_list):
            if arch_seq in seen_arch_seq:
                y[i] = 0
            else:
                seen_arch_seq.append(arch_seq)

        y = list(np.cumsum(y))
        plt.plot([0]+x, [0]+y, label=EXPNAME_TO_LABEL[exp_name])

    plt.ylabel(f"Arch > {base_r2:.2f}")
    plt.xlabel("Time (Hour)")
    plt.xlim(TMIN, TMAX)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))
    ax.xaxis.set_major_formatter(hour_major_formatter)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(150))

    plt.legend(fontsize=legend_font_size, ncol=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# def moving_average(x, w):
#     # x = np.concatenate([np.array([x[0] for i in range(w-1)]), x])
#     return np.convolve(x, np.ones(w), 'same') / w


def plot_best_networks(data, baseline_data, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    plt.plot(
        list(range(len(baseline_data[METRIC]))),
        baseline_data[METRIC],
        "k--",
        label="baseline",
    )

    for exp_name, exp_data in data.items():

        run_data = exp_data[0]
        plt.plot(
            list(range(len(run_data[METRIC]))),
            run_data[METRIC],
            label=EXPNAME_BEST_TO_LABEL[exp_name],
        )

    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.xlabel("Epochs")
    plt.ylim(*METRIC_LIMITS)

    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))
    # ax.xaxis.set_major_formatter(hour_major_formatter)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))

    plt.legend(fontsize=legend_font_size, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_training_time(data, baseline_data, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    labels = ["baseline"]
    t_times = [baseline_data["training_time"][-1]]

    for exp_name, exp_data in data.items():

        run_data = exp_data[0]
        t_times.append(run_data["training_time"])
        labels.append("\n".join(exp_name.split("_")[1:]))

    plt.bar(labels, t_times)
    plt.ylabel("Training Time (Sec.)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_time_to_solution(experiments, output_path, baseline_data=None):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    if baseline_data is None:
        return
    solution_to_reach = max(baseline_data[METRIC])

    def hash(x):
        return "_".join([str(xi) for xi in x])

    for exp_name, exp_data in experiments.items():
        t0 = exp_data["t0"]
        x = [d["time"].timestamp() - t0.timestamp() for d in exp_data["history"] if d[METRIC][-1] > solution_to_reach]
        y = [1 for d in exp_data["history"] if d[METRIC][-1] > solution_to_reach]

        if len(x) > 0:
            x, y = zip(*sorted(zip(x, y), key=lambda t: t[0]))
        else:
            x, y = [], []
        x, y = list(x), list(y)

        time_to_solution = x
        # print(time_to_solution)
        if len(time_to_solution) == 0:
            time_to_solution = TMAX
        else:
            time_to_solution = x[0]
        print(
            EXPNAME_TO_LABEL[exp_name], " -> ", f"{time_to_solution/60:.2f} min to {solution_to_reach:.3f}"
        )

        # Training time best arch
        i_max = exp_data["results"].objective.idxmax()
        if "arch_seq" in exp_data["results"].columns:
            arch_seq = hash(json.loads(exp_data["results"].iloc[i_max].arch_seq))
        else:
            arch_seq = hash(exp_data["results"].iloc[i_max].to_numpy()[:-2].astype(int))
        for hist in exp_data["history"]:
            if hash(hist["arch_seq"]) == arch_seq:
                training_time = hist["training_time"]
                n_epochs = len(hist["val_loss"])
                break
        print(f"Training time best arch ({EXPNAME_TO_LABEL[exp_name]}): {training_time/60:.2f} min. and {training_time/60/n_epochs:.2f} min/epoch")

def plot_scaling_number_of_evaluations(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exp_names = {exp_name.split("_")[-1] for exp_name in experiments}
    data_plot = {exp_name: ([], []) for exp_name in exp_names}

    for exp_name, exp_data in experiments.items():
        alg_name = exp_name.split("_")[-1]
        n_gpus = int(exp_name.split("_")[1][:-3])
        n_evals = len(exp_data["history"])

        data_plot[alg_name][0].append(n_gpus)
        data_plot[alg_name][1].append(n_evals)

    for algo, (x, y) in data_plot.items():
        plt.plot(x, y, label=algo, marker="o")

    plt.xscale("log", base=2)
    # plt.yscale("log", base=2)

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return str(int(x))

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.FixedLocator(x))
    ax.xaxis.set_major_formatter(major_formatter)

    plt.ylabel("#Evaluations")
    plt.xlabel("#GPUs per Evaluation")
    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scaling_best_objective(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exp_names = {exp_name.split("_")[-1] for exp_name in experiments}
    data_plot = {exp_name: ([], []) for exp_name in exp_names}

    for exp_name, exp_data in experiments.items():
        alg_name = exp_name.split("_")[-1]
        n_gpus = int(exp_name.split("_")[1][:-3])
        best_objective = exp_data["results"].objective.max()

        data_plot[alg_name][0].append(n_gpus)
        data_plot[alg_name][1].append(best_objective)

    for algo, (x, y) in data_plot.items():
        plt.plot(x, y, label=algo, marker="o")

    plt.xscale("log", base=2)

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return str(int(x))

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.FixedLocator(x))
    ax.xaxis.set_major_formatter(major_formatter)

    plt.ylabel(METRIC_TO_LABEL[METRIC])
    plt.xlabel("#GPUs per Evaluation")
    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scaling_time_to_solution(experiments, output_path, baseline_data=None):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    if baseline_data is None:
        return
    solution_to_reach = max(baseline_data[METRIC])

    plt.figure()

    exp_names = {exp_name.split("_")[-1] for exp_name in experiments}
    data_plot = {exp_name: ([], []) for exp_name in exp_names}

    for exp_name, exp_data in experiments.items():
        alg_name = exp_name.split("_")[-1]
        n_gpus = int(exp_name.split("_")[1][:-3])
        time_to_solution = exp_data["results"][
            exp_data["results"].objective >= solution_to_reach
        ]
        if len(time_to_solution) == 0:
            time_to_solution = 3600 * 3
        else:
            time_to_solution = time_to_solution.elapsed_sec.iloc[0]

        data_plot[alg_name][0].append(n_gpus)
        data_plot[alg_name][1].append(time_to_solution)

    for algo, (x, y) in data_plot.items():
        plt.plot(x, y, label=algo, marker="o")

    # plt.yscale("log", base=10)
    plt.xscale("log", base=2)

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return str(int(x))

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.FixedLocator(x))
    ax.xaxis.set_major_formatter(major_formatter)

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x/3600:.2f}"

    ax.yaxis.set_major_locator(ticker.MultipleLocator(3600 / 2))
    ax.yaxis.set_major_locator(
        ticker.FixedLocator([0] + [3600 / 8 * (2 ** i) for i in range(5)] + [3600 * 3])
    )
    ax.yaxis.set_major_formatter(major_formatter)

    plt.ylabel("Time To Solution (Hours)")
    plt.xlabel("#GPUs per Evaluation")
    plt.ylim(300, 3600 * 3)
    plt.grid()
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scaling_training_time(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exp_names = {exp_name.split("_")[-1] for exp_name in experiments}
    data_plot = {exp_name: ([], []) for exp_name in exp_names}

    for exp_name, exp_data in experiments.items():
        alg_name = exp_name.split("_")[-1]
        n_gpus = int(exp_name.split("_")[1][:-3])
        training_time = [d["training_time"] for d in exp_data["history"]]

        data_plot[alg_name][0].append(n_gpus)
        data_plot[alg_name][1].append(training_time)

    i = 1
    for algo, (x, y) in data_plot.items():
        plt.subplot(1, len(data_plot), i)
        plt.boxplot(
            x=y,
            positions=x,
            labels=[f"{algo}{xi}" for xi in x],
            widths=[0.5 * xi for xi in x],
        )

        plt.xscale("log", base=2)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return str(int(x))

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.xaxis.set_major_formatter(major_formatter)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return f"{x/60:.2f}"

        # ax.yaxis.set_major_locator(ticker.MultipleLocator(3600/2))
        ax.yaxis.set_major_formatter(major_formatter)

        plt.ylabel("Training Time (Min.)")
        plt.xlabel(f"#GPUs per Eval\n{algo}")
        plt.ylim(0, 4500)
        plt.grid()
        i += 1

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scaling_n_parameters(experiments, output_path):
    output_file_name = f"{inspect.stack()[0][3]}.{FILE_EXTENSION}"
    output_path = os.path.join(output_path, output_file_name)

    plt.figure()

    exp_names = {exp_name.split("_")[-1] for exp_name in experiments}
    data_plot = {exp_name: ([], []) for exp_name in exp_names}

    for exp_name, exp_data in experiments.items():
        alg_name = exp_name.split("_")[-1]
        n_gpus = int(exp_name.split("_")[1][:-3])
        training_time = [d["n_parameters"] for d in exp_data["history"]]
        print(EXPNAME_TO_LABEL[exp_name], " -> ", f"{int(np.mean(training_time)/1e6)}", " param")

        data_plot[alg_name][0].append(n_gpus)
        data_plot[alg_name][1].append(training_time)

    i = 1
    for algo, (x, y) in data_plot.items():
        plt.subplot(1, len(data_plot), i)
        plt.boxplot(
            x=y,
            positions=x,
            labels=[f"{algo}{xi}" for xi in x],
            widths=[0.5 * xi for xi in x],
        )
        # plt.hist(x=y, label=)

        plt.xscale("log", base=2)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return str(int(x))

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.xaxis.set_major_formatter(major_formatter)

        # @ticker.FuncFormatter
        # def major_formatter(x, pos):
        #     return f"{x/60:.2f}"
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(3600/2))
        # ax.yaxis.set_major_formatter(major_formatter)

        plt.ylabel("#Parameters")
        plt.xlabel(f"#GPUs per Eval\n{algo}")
        # plt.ylim(top=1e7)
        # plt.yscale("log")
        plt.grid()
        i += 1

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
            EXPNAME_BEST_TO_LABEL[best_name] = experiments["bests"][best_name]
            hists_path = os.path.join(module_path, "best", best_name, "logs", "history")
            hists = [name for name in os.listdir(hists_path) if "json" in name]
            bests_data[best_name] = [
                load_json(os.path.join(hists_path, h)) for h in hists
            ]

    experiment_folder = experiments.get("experiments_folder", "exp")

    # load the data for each experiment
    experiments = {exp: None for exp in experiments["experiments"]}

    for exp in experiments:
        experiment_path = os.path.join(module_path, experiment_folder, exp)
        data = load_data_from_exp(experiment_path)
        experiments[exp] = data

    plot_objective_multi(experiments, output_path, baseline_data)
    plot_best_objective(experiments, output_path)
    plot_number_of_evaluations(experiments, output_path)
    plot_scatter_objective_multi(experiments, output_path)
    plot_usage_training_time(experiments, output_path)
    plot_count_arch_better_than_baseline(experiments, output_path, baseline_data)
    plot_time_to_solution(experiments, output_path, baseline_data)
    # plot_diversity2_multi(experiments, output_path)
    plot_distribution_training_duration(experiments, output_path)

    # scaling plots
    num_workers_to_keep = 16
    n_gpus_per_node = 8
    scaling_experiments = {}
    for exp_name, exp_data in experiments.items():
        tmp = exp_name.split("_")[1:-1]
        n_gpus_per_eval, n_nodes = int(tmp[0][:-3]), int(tmp[1])
        workers_per_node = n_gpus_per_node / n_gpus_per_eval
        num_workers = workers_per_node * n_nodes
        if num_workers == num_workers_to_keep:
            scaling_experiments[exp_name] = exp_data

    if len(scaling_experiments) > 0:
        scaling_output_path = os.path.join(output_path, "scaling")
        create_dir(scaling_output_path)
        plot_scaling_number_of_evaluations(scaling_experiments, scaling_output_path)
        plot_scaling_best_objective(scaling_experiments, scaling_output_path)
        plot_scaling_time_to_solution(
            scaling_experiments, scaling_output_path, baseline_data
        )
        plot_scaling_training_time(scaling_experiments, scaling_output_path)
        plot_scaling_n_parameters(scaling_experiments, scaling_output_path)

    # plot best
    if bests_data is not None:
        plot_best_networks(bests_data, baseline_data, bests_output_path)
        plot_best_training_time(bests_data, baseline_data, bests_output_path)


def main():
    global METRIC, METRIC_LIMITS

    output_path = os.path.join(HERE, "outputs")
    create_dir(output_path)

    experiments_path = os.path.join(HERE, "experiments.yaml")
    experiments = yaml_load(experiments_path)

    for dataset in experiments:

        dataset_path = os.path.join(output_path, dataset)
        create_dir(dataset_path)

        METRIC = experiments[dataset]["metric"]
        METRIC_LIMITS = experiments[dataset].get("metric_limits", [])

        for experiment in experiments[dataset]["experiments"]:

            EXPNAME_TO_LABEL[experiment] = experiments[dataset]["experiments"].get(
                experiment, experiment
            )

            experiment_path = os.path.join(dataset_path, experiment)
            create_dir(experiment_path)

            experiment_folder = experiments[dataset].get("experiments_folder", "exp")
            generate_plot_from_exp(
                dataset, experiment, experiment_folder, output_path=experiment_path
            )

        # comparative plots between experiments of the same dataset
        generate_plot_from_dataset(
            dataset, experiments[dataset], output_path=dataset_path
        )


if __name__ == "__main__":
    main()
