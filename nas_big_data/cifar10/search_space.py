import collections

import tensorflow as tf

from deephyper.search.nas.model.space import AutoKSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode, MimeNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.op1d import Dense, Identity

# from deephyper.search.nas.model.space.op.cnn import
from deephyper.search.nas.model.space.op.cnn import Convolution2D


def swish(x):
    return x * tf.nn.sigmoid(x)


def add_dense_to_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case

    activations = [None, swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(
    input_shape=(32, 32, 3),
    output_shape=(10,),
    num_blocks=5,
    num_cells=2,
    *args,
    **kwargs,
):

    ss = AutoKSearchSpace(input_shape, output_shape, regression=False)
    source = prev_input = ss.input_nodes[0]

    # look over skip connections within a range of the 3 previous nodes
    # anchor_points = collections.deque([source], maxlen=3)

    node = ConstantNode(
        op=Convolution2D(filters=8, kernel_size=(3, 3), strides=1, padding="same")
    )
    ss.connect(source, node)

    # for _ in range(num_layers):
    #     vnode = VariableNode()
    #     add_dense_to_(vnode)

    #     arch.connect(prev_input, vnode)

    #     # * Cell output
    #     cell_output = vnode

    #     cmerge = ConstantNode()
    #     cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

    #     for anchor in anchor_points:
    #         skipco = VariableNode()
    #         skipco.add_op(Tensor([]))
    #         skipco.add_op(Connect(arch, anchor))
    #         arch.connect(skipco, cmerge)

    #         prev_input = cbn
    #     else:
    #         prev_input = cmerge

    #     # ! for next iter
    #     anchor_points.append(prev_input)

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_search_space()
    ops = [random() for _ in range(search_space.num_nodes)]
    print(ops)

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")

    N = 3
    shape = (32, 32, 3)

    dummy_data = np.random.rand(N, *shape)
    y = model.predict(dummy_data)
    print(np.shape(y))


if __name__ == "__main__":
    test_create_search_space()
