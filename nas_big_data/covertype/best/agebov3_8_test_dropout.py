"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.covertype.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run
from nas_big_data.covertype.load_data import load_data
from nas_big_data.covertype.dense_skipco import create_search_space

Problem.load_data(load_data, use_test=True)

Problem.search_space(create_search_space, num_layers=10, dropout=0.2)

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.001392459853203709
config["hyperparameters"]["batch_size"] = 256  # 1 rank
config["hyperparameters"]["callbacks"]["ReduceLROnPlateau"] = dict(patience=4, verbose=0)
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    9,
    1,
    30,
    1,
    1,
    28,
    1,
    1,
    0,
    24,
    0,
    0,
    1,
    23,
    0,
    1,
    0,
    27,
    1,
    1,
    1,
    18,
    0,
    1,
    1,
    2,
    0,
    1,
    1,
    14,
    1,
    1,
    0,
    20,
    1,
    0,
    1,
]

run(config)
