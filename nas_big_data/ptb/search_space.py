"""
    * Tensorflow doc page for ConvLSTM:
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/ConvLSTM2D
"""
import collections

import tensorflow as tf

from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Zero
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import AddByProjecting
from deephyper.search.nas.model.space.op.op1d import Identity, Dense
from deephyper.search.nas.model.space.op.seq import LSTM, Embedding


def add_lstm_seq_(node):
    node.add_op(Identity())  # we do not want to create a layer in this case
    for units in range(16, 1025, 16):
        node.add_op(LSTM(units=units, return_sequences=True, stateful=False))


def add_embedding_(node, vocab_size=10000, num_steps=20):
    # node.add_op(Identity())
    for emb_size in range(50, 250, 10):
        node.add_op(Embedding(vocab_size, emb_size, input_length=num_steps))


def create_search_space(
    input_shape=(20,), output_shape=(20,), num_layers=5, *args, **kwargs
):
    vocab_size = 10000
    ss = KSearchSpace(input_shape, (*output_shape, vocab_size))
    source = ss.input_nodes[0]

    emb = prev_input = VariableNode()
    add_embedding_(emb, vocab_size)
    ss.connect(source, emb)

    # look over skip connections within a range of the 2 previous nodes
    anchor_points = collections.deque([emb], maxlen=3)

    for _ in range(num_layers):
        vnode = VariableNode()
        add_lstm_seq_(vnode)

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

        # ! for next iter
        prev_input = cmerge
        anchor_points.append(prev_input)

    # out = ConstantNode(
    #     op=tf.keras.layers.TimeDistributed(
    #         tf.keras.layers.Dense(units=vocab_size, activation="softmax")
    #     )
    # )
    out = ConstantNode(op=tf.keras.layers.Dense(units=vocab_size, activation="softmax"))
    ss.connect(prev_input, out)

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_search_space(num_layers=4)
    ops = [random() for _ in range(search_space.num_nodes)]
    print(ops)

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")

    N = 3
    shape = (20,)

    dummy_data = np.random.rand(N, *shape)
    y = model.predict(dummy_data)
    print(np.shape(y))


if __name__ == "__main__":
    test_create_search_space()
