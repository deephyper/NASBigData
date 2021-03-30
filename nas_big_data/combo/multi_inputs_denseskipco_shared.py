import collections

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode, MirrorNode, MimeNode
from deephyper.nas.space.op.basic import Zero, Tensor
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting, Concatenate
from deephyper.nas.space.op.op1d import Dense, Identity, Dropout


class MultiInputsDenseSkipCoFactory(SpaceFactory):
    def build(
        self,
        input_shape,
        output_shape,
        regression=True,
        num_layers=10,
        **kwargs,
    ):
        self.ss = AutoKSearchSpace(input_shape, output_shape, regression=regression)
        self.shapes_to_vnodes = {}

        sub_graphs_outputs = []

        for input_ in self.ss.input_nodes:
            output_sub_graph = self.build_sub_graph(input_)
            sub_graphs_outputs.append(output_sub_graph)

        cmerge = ConstantNode()
        cmerge.set_op(Concatenate(self.ss, sub_graphs_outputs))

        output_sub_graph = self.build_sub_graph(cmerge)

        return self.ss

    def build_sub_graph(self, input_, num_layers=3):
        source = prev_input = input_

        mirror = False
        is_input = False
        if type(source) is ConstantNode:
            if type(source._op) is Tensor:
                if "input_" in source._op.tensor.name:
                    is_input = True
                    input_name = source._op.tensor.name
                    input_shape = tuple(source._op.tensor.shape[1:])
                    if self.shapes_to_vnodes.get(input_shape) is None:
                        self.shapes_to_vnodes[input_shape] = []
                    else:
                        mirror = True
                        memory = self.shapes_to_vnodes[input_shape][::-1]


        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)
        for layer_i in range(num_layers):
            if not(mirror):
                vnode = VariableNode()
                self.add_dense_to_(vnode)
                if is_input:
                    self.shapes_to_vnodes[input_shape].append(vnode)
            else:
                vnode = MirrorNode(memory.pop())

            self.ss.connect(prev_input, vnode)

            # * Cell output
            prev_node = vnode
            if layer_i == num_layers-1:
                return prev_node

            cmerge = ConstantNode()
            cmerge.set_op(Concatenate(self.ss, [prev_node]))

            for anchor in anchor_points:

                if not(mirror):
                    skipco = VariableNode()
                    if is_input:
                        self.shapes_to_vnodes[input_shape].append(skipco)
                else:
                    skipco = MimeNode(memory.pop())

                skipco.add_op(Zero())
                skipco.add_op(Connect(self.ss, anchor))

                self.ss.connect(skipco, cmerge)

            prev_input = cmerge

            # ! for next iter
            anchor_points.append(prev_input)

        return prev_input

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case

        activations = [None, tf.nn.swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
        for units in range(50, 2000, 25):
            for activation in activations:
                node.add_op(Dense(units=units, activation=activation))


def create_search_space(
    input_shape=[(8,), (10,), (10,)], output_shape=(1,), num_layers=3, **kwargs
):
    return MultiInputsDenseSkipCoFactory()(
        input_shape, output_shape, num_layers=num_layers, **kwargs
    )

if __name__ == "__main__":
    shapes = dict(input_shape=[(8,), (10,), (10,)], output_shape=(1,))
    factory = MultiInputsDenseSkipCoFactory()
    # factory.test(**shapes)
    factory.plot_model(**shapes)
    # factory.plot_space(**shapes)

    # space = create_search_space()
    # print("size : ", space.size)





# def test_create_search_space():
#     """Generate a random neural network from the search_space definition.
#     """
#     from random import random
#     from tensorflow.keras.utils import plot_model
#     import tensorflow as tf
#     import numpy as np

#     search_space = create_search_space()
#     ops = [
#         9,
#         1,
#         30,
#         1,
#         1,
#         28,
#         1,
#         1,
#         0,
#         24,
#         0,
#         0,
#         1,
#         23,
#         0,
#         1,
#         0,
#         27,
#         1,
#         1,
#         1,
#         18,
#         0,
#         1,
#         1,
#         2,
#         0,
#         1,
#         1,
#         14,
#         1,
#         1,
#         0,
#         20,
#         1,
#         0,
#         1,
#     ]
#     print(ops)
#     print("Search space size: ", search_space.size)

#     print(f"This search_space needs {len(ops)} choices to generate a neural network.")

#     search_space.set_ops(ops)

#     model = search_space.create_model()
#     model.summary()

#     plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
#     print("The sampled_neural_network.png file has been generated.")

#     # N = 3
#     # shape = (32, 32, 3)

#     # dummy_data = np.random.rand(N, *shape)
#     # y = model.predict(dummy_data)
#     # print(np.shape(y))


# if __name__ == "__main__":
#     test_create_search_space()
