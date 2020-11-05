import collections
import json
import os
import copy

import numpy as np
import skopt.space as space
from skopt import Optimizer as SkOptimizer
from skopt.learning import RandomForestRegressor

from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature
from deephyper.evaluator.evaluate import Encoder
from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch

dhlogger = util.conf_logger("deephyper.search.nas.bo_hpo_nas")

# def key(d):
#     return json.dumps(dict(arch_seq=d['arch_seq']), cls=Encoder)


class BoHpoNas(NeuralArchitectureSearch):
    """Aging evolution with Bayesian Optimization.

    This algorithm build on the 'Regularized Evolution' from https://arxiv.org/abs/1802.01548. It cumulates Hyperparameter optimization with bayesian optimisation and Neural architecture search with regularized evolution.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
        population_size (int, optional): the number of individuals to keep in the population. Defaults to 100.
        sample_size (int, optional): the number of individuals that should participate in each tournament. Defaults to 10.
    """

    def __init__(self, problem, run, evaluator, **kwargs):
        super().__init__(problem=problem, run=run, evaluator=evaluator, **kwargs)

        self.free_workers = self.evaluator.num_workers

        dhlogger.info(
            jm(
                type="start_infos",
                alg="bayesian-optimization-for-hpo-nas",
                nworkers=self.evaluator.num_workers,
                encoded_space=json.dumps(self.problem.space, cls=Encoder),
            )
        )

        # Setup
        self.pb_dict = self.problem.space
        cs_kwargs = self.pb_dict["create_search_space"].get("kwargs")
        if cs_kwargs is None:
            search_space = self.pb_dict["create_search_space"]["func"]()
        else:
            search_space = self.pb_dict["create_search_space"]["func"](**cs_kwargs)

        self.space_list = [
            (0, vnode.num_ops - 1) for vnode in search_space.variable_nodes
        ]

        # Initialize Hyperaparameter space
        self.dimensions = []
        self.size_ha = None  # Number of algorithm hyperparameters in the dimension list
        self.add_ha_dimensions()
        self.add_hm_dimensions()

        # Initialize opitmizer of hyperparameter space
        # acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}  # tiny exploration
        acq_func_kwargs = {"xi": 0.000001, "kappa": 1.96}  # tiny exploration
        self.n_initial_points = self.free_workers

        self.opt = SkOptimizer(
            dimensions=self.dimensions,
            base_estimator=RandomForestRegressor(n_jobs=32),
            # base_estimator=RandomForestRegressor(n_jobs=4),
            acq_func="LCB",
            acq_optimizer="sampling",
            acq_func_kwargs=acq_func_kwargs,
            n_initial_points=self.n_initial_points,
            # model_queue_size=100,
        )

    def add_ha_dimensions(self):
        """Add algorithm hyperparameters to the dimension list.
        """
        self.dimensions.append(self.problem.space["hyperparameters"]["learning_rate"])
        self.dimensions.append(self.problem.space["hyperparameters"]["batch_size"])
        self.dimensions.append(self.problem.space["hyperparameters"]["ranks_per_node"])
        self.size_ha = len(self.dimensions)

    def add_hm_dimensions(self):
        """Add model hyperparameters to the dimension list.
        """
        for low, high in self.space_list:
            self.dimensions.append(space.Integer(low, high))

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        add_arguments_from_signature(parser, BoHpoNas)
        return parser

    def saved_keys(self, val: dict):
        res = {
            "learning_rate": val["hyperparameters"]["learning_rate"],
            "batch_size": val["hyperparameters"]["batch_size"],
            "ranks_per_node": val["hyperparameters"]["ranks_per_node"],
            "arch_seq": str(val["arch_seq"]),
        }
        return res

    def main(self):

        num_evals_done = 0

        # Filling available nodes at start
        self.evaluator.add_eval_batch(self.gen_random_batch(size=self.free_workers))

        # Main loop
        while num_evals_done < self.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())

            if len(new_results) > 0:
                stats = {"num_cache_used": self.evaluator.stats["num_cache_used"]}
                dhlogger.info(jm(type="env_stats", **stats))
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    h = [
                        cfg["hyperparameters"]["learning_rate"],
                        cfg["hyperparameters"]["batch_size"],
                        cfg["hyperparameters"]["ranks_per_node"],
                    ]
                    h.extend(cfg["arch_seq"])
                    opt_X.append(h)
                    opt_y.append(-obj)

                self.opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self.opt.ask(n_points=len(new_results))

                new_batch = []
                for x in new_X:
                    new_cfg = copy.deepcopy(self.pb_dict)

                    new_cfg["hyperparameters"]["learning_rate"] = x[0]
                    new_cfg["hyperparameters"]["batch_size"] = x[1]
                    new_cfg["hyperparameters"]["ranks_per_node"] = x[2]

                    new_cfg["arch_seq"] = x[self.size_ha :]

                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self.evaluator.add_eval_batch(new_batch)

    def select_parent(self, sample: list) -> list:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg["arch_seq"]

    def gen_random_batch(self, size: int) -> list:
        batch = []

        points = self.opt.ask(n_points=size)
        for point in points:
            #! DeepCopy is critical for nested lists or dicts
            cfg = copy.deepcopy(self.pb_dict)

            # hyperparameters
            cfg["hyperparameters"]["learning_rate"] = point[0]
            cfg["hyperparameters"]["batch_size"] = point[1]
            cfg["hyperparameters"]["ranks_per_node"] = point[2]

            # architecture DNA
            cfg["arch_seq"] = point[self.size_ha :]
            batch.append(cfg)

        return batch


if __name__ == "__main__":
    args = BoHpoNas.parse_args()
    search = BoHpoNas(**vars(args))
    search.main()
