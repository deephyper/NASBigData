from deephyper.problem import NaProblem
from nas_big_data.combo.search_space import create_search_space
from nas_big_data.combo.load_data import load_data

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=5)

Problem.hyperparameters(
    lsr_batch_size=True,
    lsr_learning_rate=True,
    batch_size=32,
    learning_rate=0.001,
    optimizer="adam",
    num_epochs=100,
    verbose=0,
    callbacks=dict(
        # CSVExtendedLogger=dict(),
        ReduceLROnPlateau=dict(monitor="val_r2", mode="max", verbose=0, patience=5),
        EarlyStopping=dict(
            monitor="val_r2", min_delta=0, mode="max", verbose=0, patience=10
        ),
        )
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)