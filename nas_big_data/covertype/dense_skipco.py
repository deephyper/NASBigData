import collections

import tensorflow as tf

from deephyper.search.nas.model.space import AutoKSessSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode, MimeNode
from deephyper.search.nas.model.space.op.basic import Tensor, Zero
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.op1d import Dense, Identity, Dropout


def swish(x):
    return x * tf.nn.sigmoid(x)


def add_dense_to_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case

    activations = [None, swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
    for units in range(16, 97, 16):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(
    input_shape=(10,), output_shape=(7,), num_layers=10, dropout=0.0, *args, **kwargs
):

    regression = False
    ss = AutoKSessSpace(input_shape, output_shape, regression=regression)
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
