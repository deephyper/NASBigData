"""
KMP_BLOCK_TIME=0

horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

import os

from deephyper.nas.run.tf_distributed import run
from deephyper.nas.run.util import create_dir
from nas_big_data.combo.load_data import load_data_test
from nas_big_data.combo.problem_agebo import Problem

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(4)])

HERE = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(HERE, __file__[:-3])
create_dir(output_dir)

Problem.load_data(load_data_test)
config = Problem.space

config["log_dir"] = output_dir
config["hyperparameters"]["num_epochs"] = 200
config["hyperparameters"]["verbose"] = 1
config["hyperparameters"]["learning_rate"] = 0.0002100091516029
config["hyperparameters"]["batch_size"] = 179
config["hyperparameters"]["optimizer"] = "nadam"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 4
config["hyperparameters"]["patience_EarlyStopping"] = 13
config["loss"] = "mae"


config["arch_seq"] = [320, 0, 377, 1, 0, 129, 100, 1, 366, 1, 1, 267, 286, 1, 331, 0, 1, 338]

run(config)
