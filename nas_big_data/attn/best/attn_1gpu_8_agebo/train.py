"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

from nas_big_data.attn.problem_agebo import Problem
from nas_big_data.attn.load_data import load_data_h5
from deephyper.nas.run.tf_distributed import run
from deephyper.nas.run.util import create_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(4)])

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
    monitor="val_aucpr",
    mode="max",
    save_best_only=True,
    verbose=0,
    save_weights_only=False,
)


# Search
config["loss"] = "categorical_crossentropy"
config["hyperparameters"]["learning_rate"] = 0.0019504518
config["hyperparameters"]["batch_size"] = 553
config["hyperparameters"]["optimizer"] = "adagrad"
config["hyperparameters"]["patience_ReduceLROnPlateau"] = 5
config["hyperparameters"]["patience_EarlyStopping"] = 18
config["arch_seq"] = [119, 0, 370, 1, 0, 267, 1, 1, 1, 118, 1, 44, 0, 1, 43, 1, 1, 1]

run(config)

X_test, y_test = load_data_h5("test")

model = tf.keras.models.load_model(f"best_model_{fname}.h5")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_label = np.argmax(y_test, axis=1)
bal_acc = balanced_accuracy_score(y_test_label, y_pred)

print(confusion_matrix(y_test_label, y_pred))
print(classification_report(y_test_label, y_pred))

score = model.evaluate(X_test, y_test)
score.append(bal_acc)
score_names = ["loss", "acc", "auroc", "aucpr", "bal_acc"]
print("score:")
output = " ".join([f"{sn}:{sv:.3f}" for sn,sv in zip(score_names, score)])
print(output)
