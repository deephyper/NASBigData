"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

import os

from nas_big_data.combo.problem_ae_1 import Problem
from nas_big_data.combo.load_data import load_data_test
from deephyper.nas.run.alpha import run
from deephyper.nas.run.util import create_dir

HERE = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(HERE, __file__[:-3])
create_dir(output_dir)

Problem.load_data(load_data_test)
config = Problem.space

config["log_dir"] = output_dir
config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["verbose"] = 1
# config["hyperparameters"]["learning_rate"] = 0.00333302975
# config["hyperparameters"]["batch_size"] = 256


config["arch_seq"] = [238, 1, 261, 1, 0, 130, 1, 0, 1, 125, 1, 98, 0, 1, 263, 0, 1, 1, 298, 1, 164, 1, 0, 288, 1, 1, 0, 198, 0, 255, 0, 1, 87, 0, 0, 0]

run(config)
