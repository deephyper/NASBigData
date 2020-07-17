"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.daymet.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 10
config["hyperparameters"]["learning_rate"] = 0.001
config["hyperparameters"]["batch_size"] = 1
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    0.9154550524498652,
    0.18484514562145582,
    0.6995903612257909,
    0.4358133541183399,
    0.019096009170665784,
]

run(config)
