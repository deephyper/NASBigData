import os

import numpy as np

from nas_big_data.attn.load_data import load_data_h5
import autosklearn.classification
from sklearn.metrics import  precision_recall_curve, auc, accuracy_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=160,
    per_run_time_limit=30,
    tmp_folder=os.path.join(HERE, 'autosklearn_classification_example_tmp'),
    output_folder=os.path.join(HERE, 'autosklearn_classification_example_out'),
    memory_limit = 100 * 1024 # 50 GB
)

X_train, y_train = load_data_h5(split="train")
y_train = np.argmax(y_train, axis=1)

automl.fit(X_train, y_train, dataset_name='attn')

X_test, y_test = load_data_h5(split="test")
y_test = np.argmax(y_test, axis=1)

y_pred = automl.predict(X_test)

acc = accuracy_score(y_pred, y_test)

fpr, tpr, thresholds = roc_curve(y_pred, y_test)
roc_auc = auc(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(y_pred, y_test)
pr_auc = auc(recall, precision)

print(f"Test - acc: {acc:.3f}, auroc: {roc_auc:.3f}, aucpr: {pr_auc:.3f}")


