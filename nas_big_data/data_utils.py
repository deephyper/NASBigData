def convert_to_dataframe(X, y):
    """Convert a X(inputs), y(outputs) dataset to a dataframe format.

    Args:
        X (np.array): inputs.
        y (np.array): labels.

    Returns:
        DataFrame: a Pandas Dataframe with inputs named as "xi" and labels named as "label".
    """
    import numpy as np
    import pandas as pd

    data = np.concatenate([X, np.argmax(y, axis=1).reshape(-1, 1)], axis=1)
    df_train = pd.DataFrame(
        data=data_train,
        columns=[f"x{i}" for i in range(np.shape(data_train)[-1] - 1)] + ["label"],
    )

    return df_train
