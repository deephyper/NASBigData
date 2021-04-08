import os

import numpy as np

from nas_big_data.combo.load_data import load_data, load_data_npz_gz
import autosklearn.regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=160,
    per_run_time_limit=30,
    tmp_folder=os.path.join(HERE, 'autosklearn_regression_example_tmp'),
    output_folder=os.path.join(HERE, 'autosklearn_regression_example_out'),
    memory_limit = 100 * 1024 # 100 GB
)

(X_train, y_train), _ = load_data()

X_train = np.concatenate(X_train, axis=1)

automl.fit(X_train, y_train, dataset_name='combo')

X_test, y_test = load_data_npz_gz(test=True)
X_test = np.concatenate(X_test, axis=1)

y_pred = automl.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test - mse: {mse:.3f}, mae: {mae:.3f}, r2: {r2:.3f}")


