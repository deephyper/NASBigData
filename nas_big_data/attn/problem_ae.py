from deephyper.problem import NaProblem
from nas_big_data.attn.search_space import create_search_space
from nas_big_data.attn.load_data import load_data_cache

Problem = NaProblem(seed=2019)

Problem.load_data(load_data_cache)

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
        ReduceLROnPlateau=dict(monitor="val_aucpr", mode="max", verbose=0, patience=5),
        EarlyStopping=dict(
            monitor="val_aucpr", min_delta=0, mode="max", verbose=0, patience=10
        ),
        )
)

Problem.loss(
    "categorical_crossentropy",
    class_weights={0: 0.5186881480859765, 1: 13.877462488516892}
)

Problem.metrics(["acc", "auroc", "aucpr"])

Problem.objective("val_aucpr")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)