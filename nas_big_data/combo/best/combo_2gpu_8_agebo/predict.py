import os
import numpy as np
import tensorflow as tf

from nas_big_data.combo.load_data import load_data_npz_gz
from deephyper.nas.run.util import create_dir
from deephyper.nas.train_utils import selectMetric

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(4)])

HERE = os.path.dirname(os.path.abspath(__file__))
fname = HERE.split("/")[-1]
output_dir = "logs"
create_dir(output_dir)

X_test, y_test = load_data_npz_gz(test=True)

dependencies = {
     "r2":selectMetric("r2")
}

model = tf.keras.models.load_model(f"best_model_{fname}.h5", custom_objects=dependencies)
model.compile(
    metrics=["mse", "mae", selectMetric("r2")]
)


score = model.evaluate(X_test, y_test)
score_names = ["loss", "mse", "mae", "r2"]
print("score:")
output = " ".join([f"{sn}:{sv:.3f}" for sn,sv in zip(score_names, score)])
print(output)