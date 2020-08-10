"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.airlines.train
"""
import os
import shutil
import tensorflow as tf
import pathlib

from nas_big_data.airlines.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run
from nas_big_data.airlines.load_data import load_data
from nas_big_data.airlines.dense_skipco import create_search_space

Problem.load_data(load_data, use_test=True)

Problem.search_space(create_search_space, num_layers=10)

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.001474201683159716
config["hyperparameters"]["batch_size"] = 64  # 2 ranks
config["hyperparameters"]["callbacks"]["ReduceLROnPlateau"] = dict(patience=4, verbose=0)
config["hyperparameters"]["callbacks"]["EarlyStopping"] = dict(
    monitor="val_acc", min_delta=0, mode="max", verbose=0, patience=5
)
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    3,
    0,
    13,
    1,
    1,
    28,
    1,
    0,
    0,
    26,
    1,
    1,
    0,
    18,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    12,
    0,
    0,
    1,
    8,
    0,
    0,
    1,
    11,
    0,
    0,
    0,
    3,
    1,
    0,
    0,
]

REP = 5
seeds = [59950, 65837, 2339, 40409, 46235]

dir_name = os.path.basename(__file__)[:-3]
print(f"DIRNAME = {dir_name}")

pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

for rep in range(REP):
    config["seed"] = seeds[rep]
    run(config)
    shutil.move("training.csv", os.path.join(dir_name, f"training_{rep}.csv"))

