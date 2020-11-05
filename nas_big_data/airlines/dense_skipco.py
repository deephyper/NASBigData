import collections

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode, MimeNode
from deephyper.nas.space.op.basic import Tensor, Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Dense, Identity, Dropout


def swish(x):
    return x * tf.nn.sigmoid(x)


def add_dense_to_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case

    activations = [None, swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(
    input_shape=(7,), output_shape=(2,), num_layers=10, dropout=0.0, *args, **kwargs
):

    regression = False
    ss = AutoKSearchSpace(input_shape, output_shape, regression=regression)
    source = prev_input = ss.input_nodes[0]

    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=3)

    for _ in range(num_layers):
        vnode = VariableNode()
        add_dense_to_(vnode)

        ss.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(ss, [cell_output], activation="relu"))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Zero())
            skipco.add_op(Connect(ss, anchor))
            ss.connect(skipco, cmerge)

        prev_input = cmerge

        # ! for next iter
        anchor_points.append(prev_input)

    if dropout >= 0.0:
        dropout_node = ConstantNode(op=Dropout(rate=dropout))
        ss.connect(prev_input, dropout_node)

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_search_space()
    ops = [
        9,
        1,
        30,
        1,
        1,
        28,
        1,
        1,
        0,
        24,
        0,
        0,
        1,
        23,
        0,
        1,
        0,
        27,
        1,
        1,
        1,
        18,
        0,
        1,
        1,
        2,
        0,
        1,
        1,
        14,
        1,
        1,
        0,
        20,
        1,
        0,
        1,
    ]
    print(ops)
    print("Search space size: ", search_space.size)

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")

    # N = 3
    # shape = (32, 32, 3)

    # dummy_data = np.random.rand(N, *shape)
    # y = model.predict(dummy_data)
    # print(np.shape(y))


if __name__ == "__main__":
    test_create_search_space()
