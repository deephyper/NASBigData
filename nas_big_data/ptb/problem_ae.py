from deephyper.problem import NaProblem
from nas_big_data.ptb.search_space import create_search_space
from nas_big_data.ptb.load_data import load_data

Problem = NaProblem(seed=2019)

Problem.load_data(load_data, debug=True)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks=dict(CSVExtendedLogger=dict()),
)

Problem.loss("sparse_categorical_crossentropy")

Problem.metrics(["sparse_perplexity"])

Problem.objective("-val_sparse_perplexity")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])
