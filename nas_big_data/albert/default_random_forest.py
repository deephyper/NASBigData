import json
import os

from nas_big_data.albert.load_data import load_data
from nas_big_data import RANDOM_STATE


def run():
    """Test data with RandomForest

    accuracy_score on Train:  1.0
    accuracy_score on Test:  0.6638969849425279
    """
    from sklearn.ensemble import RandomForestClassifier

    (X_train, y_train), (X_test, y_test) = load_data(use_test=True, out_ohe=False)

    model_kwargs = dict(n_jobs=6, random_state=RANDOM_STATE)
    model = RandomForestClassifier(**model_kwargs)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(type(model).__name__)
    scores = dict(
        model=type(model).__name__,
        model_args=model_kwargs,
        train_acc=train_acc,
        test_acc=test_acc,
    )

    # save scores to disk
    fname = os.path.basename(__file__)[:-3]
    fpath = os.path.join(os.path.dirname(__file__), fname + ".json")
    with open(fpath, "w") as fp:
        json.dump(scores, fp)


if __name__ == "__main__":
    run()
