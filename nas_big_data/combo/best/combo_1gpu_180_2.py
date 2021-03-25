"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

import os

from nas_big_data.combo.problem_ae_1 import Problem
from deephyper.nas.run.alpha import run

HERE = os.path.dirname(os.path.abspath(__file__))

config = Problem.space

config["log_dir"] = HERE
config["hyperparameters"]["num_epochs"] = 200
# config["hyperparameters"]["learning_rate"] = 0.00333302975
# config["hyperparameters"]["batch_size"] = 256


config["arch_seq"] = [347, 1, 232, 0, 0, 277, 0, 0, 0, 89, 1, 96, 0, 1, 198, 0, 0, 1, 300, 0, 228, 1, 0, 187, 1, 0, 0, 228, 0, 192, 0, 1, 369, 1, 1, 1]

run(config)
