"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.ptb.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run
from random import random

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 20.0
config["hyperparameters"]["batch_size"] = 64
config["hyperparameters"]["callbacks"].pop("TimeStopping")
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    0.3694786804965776,
    0.8210662845954018,
    0.4190542593186294,
    0.5005476522893533,
    0.10545157774556424,
    0.02589620959598149,
    0.914894531118735,
    0.10416476218329462,
    0.562132460231267,
    0.20590827768777964,
    0.2084097463117235,
    0.3019792996865882,
    0.5499012793139204,
    0.3644085907473512,
]

run(config)
