"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.cifar10.problem_agebov3 import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.001061618285394656
config["hyperparameters"]["batch_size"] = 128  # 1 rank
config["hyperparameters"]["callbacks"].pop("TimeStopping")
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    8,
    3,
    0,
    1,
    7,
    9,
    1,
    1,
    7,
    3,
    1,
    3,
    0,
    1,
    4,
    4,
    4,
    9,
    0,
    0,
    0,
    5,
    1,
    2,
    2,
    3,
    3,
    0,
    4,
    6,
    3,
    3,
]

run(config)
