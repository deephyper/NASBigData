import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time

from nas_big_data.dionis.problem_ae import Problem
from nas_big_data.dionis.load_data import load_data

_, (X_test, y_test) = load_data(use_test=True)

arch_seq = [
    0,
    0,
    27,
    1,
    0,
    27,
    1,
    0,
    1,
    24,
    1,
    0,
    0,
    9,
    1,
    0,
    0,
    24,
    1,
    1,
    1,
    10,
    1,
    0,
    0,
    5,
    0,
    1,
    1,
    26,
    1,
    1,
    0,
    0,
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
