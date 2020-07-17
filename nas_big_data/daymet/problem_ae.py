from deephyper.problem import NaProblem
from nas_big_data.daymet.conv_lstm_2d import create_conv_lstm_search_space
from nas_big_data.daymet.load_data import load_data

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_conv_lstm_search_space, num_layers=10)

Problem.hyperparameters(
    batch_size=256,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks=dict(CSVExtendedLogger=dict(), TimeStopping=dict(seconds=460)),
)

Problem.loss("mse")

Problem.metrics(["mae"])

Problem.objective("val_loss")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
