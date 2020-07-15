import numpy as np


def run(config):
    lr = config["hyperparameters"]["learning_rate"]
    bs = config["hyperparameters"]["batch_size"]
    rpn = config["hyperparameters"]["ranks_per_node"]
    return sum(config["arch_seq"]) + lr + bs + rpn
