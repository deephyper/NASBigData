"""
    * Tensorflow doc page for ConvLSTM:
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/ConvLSTM2D
"""
import collections

import tensorflow as tf

from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.op1d import Identity


def add_convlstm_to_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case
    activations = [None, tf.nn.swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for filters in range(16, 97, 16):
        for activation in activations:
            node.add_op(
                tf.keras.layers.ConvLSTM2D(
                    filters=filters,
                    kernel_size=1,
                    activation=activation,
                    padding="same",
                    return_sequences=True,
                )
            )


def add_convlstm_oplayer_(node, units):
    node.set_op(
        tf.keras.layers.ConvLSTM2D(
            filters=units,
            kernel_size=1,
            activation="linear",
            padding="same",
            return_sequences=True,
        )
    )


def create_conv_lstm_search_space(
    input_shape=(28, 28, 1, 5),
    output_shape=(28, 28, 1, 5),
    num_layers=10,
    *args,
    **kwargs,
):
    arch = KSearchSpace(input_shape, output_shape)
    source = prev_input = arch.input_nodes[0]
    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=3)
    for _ in range(num_layers):
        vnode = VariableNode()
        add_convlstm_to_(vnode)
        arch.connect(prev_input, vnode)
        # * Cell output
        cell_output = vnode
        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu", axis=-2))
        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)
        # ! for next iter
        prev_input = cmerge
        anchor_points.append(prev_input)

    # Add layer to enforce consistency
    cnode = ConstantNode()
    units = output_shape[-1]
    add_convlstm_oplayer_(cnode, units)
    arch.connect(prev_input, cnode)
    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_conv_lstm_search_space(num_layers=2)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")

    N = 3
    shape = (28, 28, 1, 5)

    dummy_data = np.random.rand(N, *shape)
    y = model.predict(dummy_data)
    print(np.shape(y))


if __name__ == "__main__":
    test_create_search_space()
