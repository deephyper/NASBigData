import os
import numpy as np
import tensorflow as tf

from nas_big_data.combo.problem_ae import Problem
from nas_big_data.combo.load_data import load_data, load_data_npz_gz
from deephyper.nas.run.alpha import run
from deephyper.nas.run.util import create_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(8)])

HERE = os.path.dirname(os.path.abspath(__file__))
fname = HERE.split("/")[-1]
output_dir = "logs"
create_dir(output_dir)

config = Problem.space

config["log_dir"] = output_dir
config["id"] = fname
config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["verbose"] = 1
config["hyperparameters"]["callbacks"]["ModelCheckpoint"] = dict(
    monitor="val_r2",
    mode="max",
    save_best_only=True,
    verbose=0,
    save_weights_only=False,
)

# Search
config["loss"] = "mae"
config["hyperparameters"]["learning_rate"] = 0.0003589939
config["hyperparameters"]["batch_size"] = 73
config["hyperparameters"]["optimizer"] = "adamax"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 3
config["hyperparameters"]["patience_EarlyStopping"] = 24
config["arch_seq"] = [34, 0, 139, 0, 1, 155, 3, 0, 324, 0, 0, 307, 283, 0, 273, 1, 1, 66]

run(config)