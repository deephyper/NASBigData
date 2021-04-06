import numpy as np
from nas_big_data.attn.load_data import load_data_h5
from sklearn.utils.class_weight import compute_class_weight

_, y_train = load_data_h5("train")

y_integers = y_train #np.argmax(y_train, axis=1)
class_weights = compute_class_weight("balanced", np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
print(class_weights)
print(d_class_weights)