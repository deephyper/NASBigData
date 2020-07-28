from deephyper.problem import NaProblem
from nas_big_data.cifar10.load_data import load_data
from nas_big_data.cifar10.search_space_darts import create_search_space

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=[32, 64, 128, 256, 512, 1024],
    learning_rate=(0.001, 0.1, "log-uniform"),
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks=dict(
        CSVExtendedLogger=dict(),
        TimeStopping=dict(seconds=1200),
        EarlyStopping=dict(
            monitor="val_acc", min_delta=0, mode="max", verbose=0, patience=5
        ),
        ReduceLROnPlateau=dict(patience=4, verbose=0),
        ModelCheckpoint=dict(
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=0,
            filepath="model.h5",
            save_weights_only=True,
        ),
    ),
)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])

Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])
