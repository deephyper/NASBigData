import collections
from itertools import cycle

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode, MimeNode
from deephyper.nas.space.op.basic import Tensor
from deephyper.nas.space.op.cnn import Conv2D, AvgPool2D, MaxPool2D, SeparableConv2D
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByPadding, Concatenate
from deephyper.nas.space.op.op1d import Dense, Identity


normal_nodes = []
cycle_normal_nodes = cycle(normal_nodes)

reduction_nodes = []
cycle_reduction_nodes = cycle(reduction_nodes)


def generate_conv_node(strides, mime=False):
    if mime:
        if strides > 1:
            node = MimeNode(next(cycle_reduction_nodes), name="Conv")
        else:
            node = MimeNode(next(cycle_normal_nodes), name="Conv")
    else:
        node = VariableNode(name="Conv")
        if strides > 1:
            reduction_nodes.append(node)
        else:
            normal_nodes.append(node)

    padding = "valid" if strides > 1 else "same"
    node.add_op(Identity())
    node.add_op(Conv2D(filters=8, kernel_size=(1, 1), strides=strides, padding=padding))
    node.add_op(Conv2D(filters=8, kernel_size=(3, 3), strides=strides, padding=padding))
    node.add_op(Conv2D(filters=8, kernel_size=(5, 5), strides=strides, padding=padding))
    node.add_op(AvgPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(MaxPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(MaxPool2D(pool_size=(5, 5), strides=strides, padding=padding))
    node.add_op(MaxPool2D(pool_size=(7, 7), strides=strides, padding=padding))
    node.add_op(
        SeparableConv2D(kernel_size=(3, 3), filters=8, strides=strides, padding=padding)
    )
    node.add_op(
        SeparableConv2D(kernel_size=(5, 5), filters=8, strides=strides, padding=padding)
    )
    node.add_op(
        SeparableConv2D(kernel_size=(7, 7), filters=8, strides=strides, padding=padding)
    )
    if strides == 1:
        node.add_op(
            Conv2D(
                filters=8,
                kernel_size=(3, 3),
                strides=strides,
                padding=padding,
                dilation_rate=2,
            )
        )
    return node


def generate_block(ss, anchor_points, strides=1, mime=False):

    # generate block
    n1 = generate_conv_node(strides=strides, mime=mime)
    n2 = generate_conv_node(strides=strides, mime=mime)
    add = ConstantNode(op=AddByPadding(ss, [n1, n2], activation=None))

    if len(anchor_points) == 1:
        source = anchor_points[0]
        ss.connect(source, n1)
        ss.connect(source, n2)
    else:
        if mime:
            if strides > 1:
                src_node = next(cycle_reduction_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
                src_node = next(cycle_reduction_nodes)
                skipco2 = MimeNode(src_node, name="SkipCo2")
            else:
                src_node = next(cycle_normal_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
                src_node = next(cycle_normal_nodes)
                skipco2 = MimeNode(src_node, name="SkipCo2")
        else:
            skipco1 = VariableNode(name="SkipCo1")
            skipco2 = VariableNode(name="SkipCo2")
            if strides > 1:
                reduction_nodes.append(skipco1)
                reduction_nodes.append(skipco2)
            else:
                normal_nodes.append(skipco1)
                normal_nodes.append(skipco2)
        for anchor in anchor_points:
            skipco1.add_op(Connect(ss, anchor))
            ss.connect(skipco1, n1)

            skipco2.add_op(Connect(ss, anchor))
            ss.connect(skipco2, n2)
    return add


def generate_cell(ss, hidden_states, num_blocks=5, strides=1, mime=False):
    anchor_points = [h for h in hidden_states]
    boutputs = []
    for _ in range(num_blocks):
        bout = generate_block(ss, anchor_points, strides=1, mime=mime)
        anchor_points.append(bout)
        boutputs.append(bout)

    concat = ConstantNode(op=Concatenate(ss, boutputs, not_connected=True))
    return concat


def create_search_space(
    input_shape=(32, 32, 3),
    output_shape=(10,),
    num_blocks=3,
    normal_cells=5,
    reduction_cells=1,
    repetitions=4,
    *args,
    **kwargs,
):

    ss = AutoKSearchSpace(input_shape, output_shape, regression=False)
    source = prev_input = ss.input_nodes[0]

    # look over skip connections within a range of the 3 previous nodes
    hidden_states = collections.deque([source, source], maxlen=2)

    for ri in range(repetitions):
        for nci in range(normal_cells):
            # generate a normal cell
            cout = generate_cell(
                ss, hidden_states, num_blocks, strides=1, mime=ri + nci > 0
            )
            hidden_states.append(cout)

        for rci in range(reduction_cells):
            # generate a reduction cell
            cout = generate_cell(
                ss, hidden_states, num_blocks, strides=2, mime=ri + rci > 0
            )
            hidden_states.append(cout)

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
    print("Search space size: ", search_space.size)

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
