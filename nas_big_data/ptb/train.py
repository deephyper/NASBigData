"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.ptb.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run
from random import random

config = Problem.space

config["hyperparameters"]["num_epochs"] = 5
config["hyperparameters"]["learning_rate"] = 0.01
config["hyperparameters"]["batch_size"] = 20
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [random() for _ in range(6)]

run(config)
