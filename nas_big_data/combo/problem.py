from deephyper.problem import NaProblem
from nas_big_data.combo.search_space import create_search_space
from nas_big_data.combo.load_data import load_data

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=5)

Problem.hyperparameters(
    batch_size=256,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks={} #dict(CSVExtendedLogger=dict()),
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])