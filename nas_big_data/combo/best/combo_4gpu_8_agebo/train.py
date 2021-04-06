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
config["loss"] = "mae"
config["hyperparameters"]["learning_rate"] = 0.0001182624
config["hyperparameters"]["batch_size"] = 142
config["hyperparameters"]["optimizer"] = "adamax"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 7
config["hyperparameters"]["patience_EarlyStopping"] = 24
config["arch_seq"] = [114, 0, 141, 1, 0, 217, 198, 1, 212, 1, 0, 96, 228, 1, 122, 1, 1, 308]

run(config)

X_test, y_test = load_data_npz_gz(test=True)

model = tf.keras.models.load_model(f"best_model_{fname}.h5")

score = model.evaluate(X_test, y_test)
score_names = ["loss", "r2"]
print("score:")
output = " ".join([f"{sn}:{sv:.3f}" for sn,sv in zip(score_names, score)])
print(output)