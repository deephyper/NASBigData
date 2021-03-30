from nas_big_data.combo.multi_inputs_denseskipco_shared import MultiInputsDenseSkipCoFactory

def create_search_space(
    input_shape=[(942,), (3839,), (3839,)], output_shape=(1,), num_layers=5, **kwargs
):
    return MultiInputsDenseSkipCoFactory()(
        input_shape, output_shape, num_layers=num_layers, **kwargs
    )

if __name__ == "__main__":
    # shapes = dict(input_shape=[(8,), (10,)], output_shape=(1,))
    # factory = MultiInputsDenseSkipCoFactory()
    # factory.test(**shapes)
    # factory.plot_model(**shapes)
    # factory.plot_space(**shapes)

    space = create_search_space()
    print(space.size)





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
