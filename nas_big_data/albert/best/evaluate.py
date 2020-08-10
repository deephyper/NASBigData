import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time

from nas_big_data.albert.problem_ae import Problem
from nas_big_data.albert.load_data import load_data

_, (X_test, y_test) = load_data(use_test=True)

arch_seq = [
    1,
    0,
    14,
    0,
    1,
    10,
    0,
    1,
    0,
    30,
    0,
    0,
    1,
    26,
    1,
    1,
    1,
    26,
    1,
    1,
    1,
    18,
    1,
    0,
    1,
    6,
    0,
    0,
    0,
    9,
    1,
    1,
    0,
    28,
    1,
    1,
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
