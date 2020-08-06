from deephyper.problem import NaProblem
from nas_big_data.covertype.dense_skipco import create_search_space
from nas_big_data.covertype.load_data import load_data

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=10)

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
