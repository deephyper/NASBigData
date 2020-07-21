"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from nas_big_data.cifar10.problem_ae import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 20
config["hyperparameters"]["learning_rate"] = 0.01
config["hyperparameters"]["batch_size"] = 32
config["hyperparameters"]["verbose"] = 1


config["arch_seq"] = [
    0.03004849794707598,
    0.27984801374270807,
    0.7903569133893682,
    0.8193542873049703,
    0.8612244316843909,
    0.9669363878871675,
]

run(config)
