import os
import json
from pprint import pprint
import time
import ray
import tensorflow as tf

from nas_big_data.combo.problem_agebo import Problem, load_data_cache


HERE = os.path.dirname(os.path.abspath(__file__))
best_arch_path = os.path.join(HERE, "exp_sc21", "best_arch")

def setup_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def compute_inference_time(arch_seq):
    setup_gpu()

    (X, _), _ = load_data_cache()

    distributed_strategy = tf.distribute.MirroredStrategy()
    n_replicas = distributed_strategy.num_replicas_in_sync
    print("Numbers of Replicas in Sync: ", n_replicas)

    with distributed_strategy.scope():
        model = Problem.get_keras_model(arch_seq)


        t1 = time.time()
        y_pred = model.predict(X)
        t2 = time.time()
        duration = t2 - t1

    return duration

def run():

    ray.init(address="auto", num_cpus=8, num_gpus=8)
    results = {}

    for f in os.listdir(best_arch_path):
        if ".json" in f:
            fpath = os.path.join(best_arch_path, f)
            with open(fpath, "r") as fb:
                arch_seq = json.load(fb)["0"]["arch_seq"]

            nr = int(f.split("_")[1][0])
            func = ray.remote(num_cpus=nr, num_gpus=nr)(compute_inference_time)
            duration = func.remote(arch_seq)
            exp_name = f.split(".")[0]
            results[exp_name] = duration
            print(f"{exp_name} -> {duration/60:.2f} min.")

    pprint(results)

    with open(os.path.join(HERE, "inference.json"), "w") as fb:
        json.dump(results, fb, indent=2)


if __name__ == "__main__":
    run()