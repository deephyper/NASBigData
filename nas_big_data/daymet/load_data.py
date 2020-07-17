"""
Data from Daymet at:
https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1328

requires:
pip install netCDF4
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    fname = os.path.join(HERE, "daymet_v3_tmax_2014_na_tmax.npy")
    # data = np.load(fname)[0:100]  # OOM when we use 365 days
    data = np.load(fname)[:20]

    # Shape data into channel based format
    num_days = np.shape(data)[0]
    window_length = 7

    print(f"Loaded {num_days} days with a window length of size {window_length}")

    input_array = np.zeros(
        shape=(
            num_days - 2 * window_length,
            window_length,
            np.shape(data)[1],
            np.shape(data)[2],
            1,
        )
    )
    output_array = np.zeros(
        shape=(
            num_days - 2 * window_length,
            window_length,
            np.shape(data)[1],
            np.shape(data)[2],
            1,
        )
    )

    # Shovel data into our arrays
    sample = 0
    for i in range(0, num_days - 2 * window_length):
        input_array[sample, :, :, :, 0] = data[i : i + window_length, :, :]
        output_array[sample, :, :, :, 0] = data[
            i + window_length : i + 2 * window_length, :, :
        ]
        sample += 1

    print("Dataset shape for ConvLSTM2D interface")
    print(np.shape(input_array))
    print(np.shape(output_array))

    return (input_array, output_array), (input_array, output_array)


if __name__ == "__main__":
    load_data()
