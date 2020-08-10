import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time

from nas_big_data.covertype.problem_ae import Problem
from nas_big_data.covertype.load_data import load_data

_, (X_test, y_test) = load_data(use_test=True)

arch_seq = [
    9,
    1,
    30,
    1,
    1,
    28,
    1,
    1,
    0,
    24,
    0,
    0,
    1,
    23,
    0,
    1,
    0,
    27,
    1,
    1,
    1,
    18,
    0,
    1,
    1,
    2,
    0,
    1,
    1,
    14,
    1,
    1,
    0,
    20,
    1,
    0,
    1,
]


model = Problem.get_keras_model(arch_seq)

model.save_weights("myweights")

t1 = time.time()
model.load_weights("myweights")
y_pred = model.predict(X_test)
t2 = time.time()

data_json = {"timing_predict": t2 - t1}
with open("timing_predict.json", "w") as fp:
    json.dump(data_json, fp)
