import json
import os
import numpy as np

import tensorflow as tf

from nas_big_data.albert.load_data import load_data

import wandb
from wandb.keras import WandbCallback

wandb.init(project="agebo")

HERE = os.path.dirname(__file__)


def run():
    (X_train, y_train), (X_test, y_test), categorical_indicator = load_data(
        use_test=True, out_ohe=True
    )

    # collect categorical variables indexes
    categorical_indexes = []
    categorical_sizes = []
    other_indexes = []
    for i, (categorical, size) in enumerate(categorical_indicator):
        if categorical:
            categorical_indexes.append(i)
            categorical_sizes.append(size)
        else:
            other_indexes.append(i)

    # model definition
    input_shape = np.shape(X_train)[1:]
    print("Input shape: ", input_shape)
    output_size = np.shape(y_train)[1]
    print("Output size: ", output_size)
    inputs = tf.keras.layers.Input(input_shape)
    print("Input tensor shape: ", inputs.shape)

    # Management of dense variables for the model
    dense_variables = []
    for i in other_indexes:
        dense_variables.append(inputs[:, i : i + 1])
    dense_variables = tf.keras.layers.Concatenate()(dense_variables)

    def dense_sub_model(x):
        n_layers = 3
        n_units = 16
        activation = "relu"
        for i in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=activation)(x)
            # x = tf.keras.layers.Dropout(0.25)(x)
        return x

    dense_out = dense_sub_model(dense_variables)

    # Management of categorical variables for the model

    def categorical_sub_model(x, size):
        n_layers = 3
        n_units = 8
        activation = "relu"
        x = tf.keras.layers.Embedding(
            size,
            8,
            # embeddings_constraint=tf.keras.constraints.unit_norm(),
            embeddings_regularizer=tf.keras.regularizers.l2(0.001),
        )(x)[:, 0]
        for i in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=activation)(x)
            # x = tf.keras.layers.Dropout(0.25)(x)
        # x = tf.keras.layers.Dense(1, activation="softplus")(x)
        return x

    categorical_variables = []
    for i, size in zip(categorical_indexes, categorical_sizes):
        categorical_input = inputs[:, i : i + 1]
        out_i = categorical_sub_model(categorical_input, size)
        categorical_variables.append(out_i)
    categorical_out = tf.keras.layers.Concatenate()(categorical_variables)
    # categorical_out = tf.keras.layers.Dense(32, activation="relu")(categorical_out)
    # categorical_out = tf.keras.layers.Dropout(0.2)(categorical_out)

    # output of the model
    outputs = tf.keras.layers.Concatenate()([dense_out, categorical_out])
    outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)
    # outputs = tf.keras.layers.Dropout(0.25)(outputs)
    outputs = tf.keras.layers.Dense(output_size, activation="softmax")(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print("Model created")
    fmodel_path = os.path.join(HERE, "default_neural_network.png")
    tf.keras.utils.plot_model(model, fmodel_path)
    # model.summary()
    params = model.count_params()
    print("Model Trainable Parameters: ", params)

    # model optimization hyperparameters
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        shuffle=True,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=[WandbCallback()],
    )

    test_scores = model.evaluate(X_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print()
    print(history.history)

    model.save(os.path.join(wandb.run.dir, "model.h5"))


# train_acc = model.score(X_train, y_train)
# test_acc = model.score(X_test, y_test)

# print(type(model).__name__)
# scores = dict(
#     model=type(model).__name__,
#     model_args=model_kwargs,
#     train_acc=train_acc,
#     test_acc=test_acc,
# )

# # save scores to disk
# fname = os.path.basename(__file__)[:-3]
# fpath = os.path.join(os.path.dirname(__file__), fname + ".json")
# with open(fpath, "w") as fp:
#     json.dump(scores, fp)


if __name__ == "__main__":
    run()
