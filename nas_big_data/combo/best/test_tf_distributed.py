"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

import os

# import ray
from deephyper.nas.run.tf_distributed import run
from deephyper.nas.run.util import create_dir
from nas_big_data.combo.load_data import load_data_test
from nas_big_data.combo.problem_agebo_test import Problem

# ray.init(address="auto")
# run = ray.remote(num_cpus=2, num_gpus=2)(run)

HERE = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(HERE, __file__[:-3])
create_dir(output_dir)

Problem.load_data(load_data_test)
config = Problem.space

config["log_dir"] = output_dir
config["hyperparameters"]["num_epochs"] = 4
config["hyperparameters"]["verbose"] = 1
config["hyperparameters"]["learning_rate"] = 0.00044295728306955007
config["hyperparameters"]["batch_size"] = 221
config["hyperparameters"]["optimizer"] = "adam"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 6
config["hyperparameters"]["patience_EarlyStopping"] = 22
config["loss"] = "huber_loss"


config["arch_seq"] = [371, 0, 194, 1, 0, 221, 0, 0, 1, 308, 1, 347, 1, 1, 306, 1, 0, 1, 101, 0, 34, 1, 0, 1, 1, 0, 0, 58, 0, 187, 0, 1, 112, 1, 0, 0]

# run.remote(config)
run(config)
