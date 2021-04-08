import os
import numpy as np
import tensorflow as tf

from nas_big_data.combo.problem_ae import Problem
from nas_big_data.combo.load_data import load_data, load_data_npz_gz
from deephyper.nas.run.alpha import run
from deephyper.nas.run.util import create_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(1)])

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
config["loss"] = "huber_loss"
config["hyperparameters"]["learning_rate"] = 0.0003280359
config["hyperparameters"]["batch_size"] = 185
config["hyperparameters"]["optimizer"] = "nadam"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 3
config["hyperparameters"]["patience_EarlyStopping"] = 14
config["arch_seq"] = [115, 0, 199, 1, 0, 217, 186, 0, 137, 0, 0, 191, 353, 1, 317, 1, 1, 312]

run(config)