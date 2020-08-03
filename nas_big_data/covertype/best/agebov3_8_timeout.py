"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.covertype.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.0015041914104249976
config["hyperparameters"]["batch_size"] = 128  # 8 ranks
config["hyperparameters"]["callbacks"]["ReduceLROnPlateau"] = dict(patience=4, verbose=0)


config["arch_seq"] = [
    22,
    0,
    11,
    0,
    0,
    17,
    0,
    0,
    0,
    27,
    0,
    0,
    0,
    27,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    29,
    0,
    0,
    0,
    11,
    1,
    1,
    1,
    17,
    1,
    0,
    1,
    16,
    1,
    0,
    0,
]

run(config)
