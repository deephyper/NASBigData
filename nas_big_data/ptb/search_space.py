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


def create_search_space(input_shape=(20,), output_shape=(20,), *args, **kwargs):

    vocab_size = 10000
    num_steps = 20
    import numpy as np

    print("output_shape -> ", np.shape(output_shape))

    ss = KSearchSpace(input_shape, (*output_shape, vocab_size))
    source = prev_input = ss.input_nodes[0]

    emb = ConstantNode()
    emb.set_op(Embedding(vocab_size, 64, input_length=num_steps))
    ss.connect(source, emb)

    node = ConstantNode()
    node.set_op(LSTM(units=10, return_sequences=True))
    ss.connect(emb, node)

    out = ConstantNode(
        op=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=vocab_size, activation="softmax")
        )
    )
    ss.connect(node, out)

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    import numpy as np

    search_space = create_search_space(num_layers=2)
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
