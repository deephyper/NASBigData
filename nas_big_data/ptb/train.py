"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.ptb.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 1
config["hyperparameters"]["learning_rate"] = 0.00333302975
config["hyperparameters"]["batch_size"] = 32
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = []

run(config)
