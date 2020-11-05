import collections
from itertools import cycle

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode, MimeNode
from deephyper.nas.space.op.basic import Tensor, Zero
from deephyper.nas.space.op.cnn import Conv2D, AvgPool2D, MaxPool2D, SeparableConv2D
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByPadding, Concatenate
from deephyper.nas.space.op.op1d import Dense, Identity, Dropout


normal_nodes = []
cycle_normal_nodes = cycle(normal_nodes)

reduction_nodes = []
cycle_reduction_nodes = cycle(reduction_nodes)


def generate_conv_node(strides, mime=False, first=False, num_filters=8):
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

    padding = "same"
    if first:
        node.add_op(Identity())
    else:
        node.add_op(Zero())
    node.add_op(Identity())
    node.add_op(
        Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=strides,
            padding=padding,
            activation=tf.nn.relu,
        )
    )
    node.add_op(
        Conv2D(
            filters=num_filters,
            kernel_size=(5, 5),
            strides=strides,
            padding=padding,
            activation=tf.nn.relu,
        )
    )
    node.add_op(AvgPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(MaxPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(
        SeparableConv2D(
            kernel_size=(3, 3), filters=num_filters, strides=strides, padding=padding
        )
    )
    node.add_op(
        SeparableConv2D(
            kernel_size=(5, 5), filters=num_filters, strides=strides, padding=padding
        )
    )
    if strides == 1:
        node.add_op(
            Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                strides=strides,
                padding=padding,
                dilation_rate=2,
            )
        )
        node.add_op(
            Conv2D(
                filters=num_filters,
                kernel_size=(5, 5),
                strides=strides,
                padding=padding,
                dilation_rate=2,
            )
        )
    return node


def generate_block(ss, anchor_points, strides=1, mime=False, first=False, num_filters=8):

    # generate block
    n1 = generate_conv_node(
        strides=strides, mime=mime, first=first, num_filters=num_filters
    )
    n2 = generate_conv_node(strides=strides, mime=mime, num_filters=num_filters)
    add = ConstantNode(op=AddByPadding(ss, [n1, n2], activation=None))

    if first:
        source = anchor_points[-1]
        ss.connect(source, n1)

    if mime:
        if strides > 1:
            if not first:
                src_node = next(cycle_reduction_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
            src_node = next(cycle_reduction_nodes)
            skipco2 = MimeNode(src_node, name="SkipCo2")
        else:
            if not first:
                src_node = next(cycle_normal_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
            src_node = next(cycle_normal_nodes)
            skipco2 = MimeNode(src_node, name="SkipCo2")
    else:
        if not first:
            skipco1 = VariableNode(name="SkipCo1")
        skipco2 = VariableNode(name="SkipCo2")
        if strides > 1:
            if not first:
                reduction_nodes.append(skipco1)
            reduction_nodes.append(skipco2)
        else:
            if not first:
                normal_nodes.append(skipco1)
            normal_nodes.append(skipco2)
    for anchor in anchor_points:
        if not first:
            skipco1.add_op(Connect(ss, anchor))
            ss.connect(skipco1, n1)

        skipco2.add_op(Connect(ss, anchor))
        ss.connect(skipco2, n2)
    return add


def generate_cell(ss, hidden_states, num_blocks=5, strides=1, mime=False, num_filters=8):
    anchor_points = [h for h in hidden_states]
    boutputs = []
    for i in range(num_blocks):
        bout = generate_block(
            ss, anchor_points, strides=1, mime=mime, first=i == 0, num_filters=num_filters
        )
        anchor_points.append(bout)
        boutputs.append(bout)

    concat = ConstantNode(op=Concatenate(ss, boutputs))
    return concat


def create_search_space(
    input_shape=(32, 32, 3),
    output_shape=(10,),
    num_filters=8,
    num_blocks=4,
    normal_cells=2,
    reduction_cells=1,
    repetitions=3,
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
                ss,
                hidden_states,
                num_blocks,
                strides=1,
                mime=ri + nci > 0,
                num_filters=num_filters,
            )
            hidden_states.append(cout)

        if ri < repetitions - 1:  # we don't want the last cell to be a reduction cell
            for rci in range(reduction_cells):
                # generate a reduction cell
                cout = generate_cell(
                    ss,
                    hidden_states,
                    num_blocks,
                    strides=2,
                    mime=ri + rci > 0,
                    num_filters=num_filters,
                )
                hidden_states.append(cout)

    # out_node = ConstantNode(op=Dense(100, activation=tf.nn.relu))
    out_dense = VariableNode()
    out_dense.add_op(Identity())
    for units in [10, 20, 50, 100, 200, 500, 1000]:
        out_dense.add_op(Dense(units, activation=tf.nn.relu))
    ss.connect(cout, out_dense)

    out_dropout = VariableNode()
    out_dropout.add_op(Identity())
    for drop_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]:
        out_dropout.add_op(Dropout(rate=drop_rate))
    ss.connect(out_dense, out_dropout)

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_search_space()
    # ops = [random() for _ in range(search_space.num_nodes)]
    # ops = [0 for _ in range(search_space.num_nodes)]
    ops = [
        2,
        6,
        0,
        1,
        1,
        2,
        2,
        3,
        0,
        2,
        2,
        1,
        0,
        4,
        1,
        7,
        8,
        0,
        5,
        0,
        2,
        2,
        6,
        3,
        1,
        1,
        1,
        5,
        2,
        0,
        0,
        7,
    ]
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
