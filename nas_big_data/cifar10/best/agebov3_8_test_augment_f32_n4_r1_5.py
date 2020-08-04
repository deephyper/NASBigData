"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.cifar10.problem_agebov3 import Problem
from nas_big_data.cifar10.load_data import load_data
from nas_big_data.cifar10.search_space_darts import create_search_space
from deephyper.search.nas.model.run.horovod import run

Problem.load_data(load_data, use_test=True)
config = Problem.space

Problem.search_space(
    create_search_space, num_filters=32, normal_cells=4, reduction_cells=1, repetitions=5
)

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.0011612519937434927
config["hyperparameters"]["batch_size"] = 64  # 1 rank
config["hyperparameters"]["callbacks"].pop("TimeStopping")
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    2,
    6,
    0,
    1,
    1,
    2,
    2,
    3,
    0,
    2,
    2,
    1,
    0,
    4,
    1,
    7,
    8,
    0,
    5,
    0,
    2,
    2,
    6,
    3,
    1,
    1,
    1,
    5,
    2,
    0,
    0,
    7,
]

run(config)
