import traceback

from deephyper.benchmark.nas.covertype.load_data import load_data
from deephyper.problem import NaProblem
from nas_big_data.cifar10.search_space import create_search_space

# from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=256,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks=dict(CSVExtendedLogger=dict()),
)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])

Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])
