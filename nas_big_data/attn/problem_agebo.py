from deephyper.problem import NaProblem
from nas_big_data.attn.search_space import create_search_space
from nas_big_data.attn.load_data import load_data_cache

Problem = NaProblem(seed=2019)

Problem.load_data(load_data_cache)

Problem.search_space(create_search_space, num_layers=5)

# schedules: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules

Problem.hyperparameters(
    lsr_batch_size=True,
    lsr_learning_rate=True,
    batch_size=Problem.add_hyperparameter((16, 2048, "log-uniform"), "batch_size"),
    learning_rate=Problem.add_hyperparameter(
        (1e-4, 0.01, "log-uniform"),
        "learning_rate",
    ),
    optimizer=Problem.add_hyperparameter(
        ["sgd", "rmsprop", "adagrad", "adam", "adadelta", "adamax", "nadam"], "optimizer"
    ),
    patience_ReduceLROnPlateau=Problem.add_hyperparameter(
        (3, 30), "patience_ReduceLROnPlateau"
    ),
    patience_EarlyStopping=Problem.add_hyperparameter((3, 30), "patience_EarlyStopping"),
    num_epochs=100,
    verbose=0,
    callbacks=dict(
        ReduceLROnPlateau=dict(monitor="val_aucpr", mode="max", verbose=0, patience=5),
        EarlyStopping=dict(
            monitor="val_aucpr", min_delta=0, mode="max", verbose=0, patience=10
        ),
    ),
)

Problem.loss("binary_crossentropy")

Problem.metrics(["acc", "auroc", "aucpr"])

Problem.objective("val_auc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
