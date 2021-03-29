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
config["hyperparameters"]["learning_rate"] = 0.0003233381265748
config["hyperparameters"]["batch_size"] = 166
config["hyperparameters"]["optimizer"] = "adam"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 10
config["hyperparameters"]["patience_EarlyStopping"] = 29
config["loss"] = "mae"


config["arch_seq"] = [68, 0, 154, 1, 1, 234, 0, 0, 0, 220, 1, 252, 0, 1, 254, 1, 0, 1, 219, 1, 212, 1, 1, 276, 0, 0, 0, 359, 1, 373, 1, 1, 157, 0, 0, 1]

run(config)
