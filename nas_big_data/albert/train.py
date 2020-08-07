"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.albert.train
"""

from nas_big_data.albert.problem_agebov3 import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.00333302975
config["hyperparameters"]["batch_size"] = 256
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    22,
    0,
    22,
    0,
    0,
    27,
    1,
    0,
    0,
    22,
    0,
    0,
    0,
    17,
    1,
    0,
    0,
    27,
    0,
    0,
    0,
    19,
    1,
    0,
    0,
    30,
    0,
    1,
    0,
    9,
    1,
    0,
    1,
    17,
    1,
    1,
    0,
]

run(config)
