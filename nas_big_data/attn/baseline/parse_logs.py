import os
import re
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():

    metrics = [
        "loss",
        "acc",
        "auc",
        "val_loss",
        "val_acc",
        "val_auc",
    ]
    data = {m: [] for m in metrics}
    data["training_time"] = []

    fname = "baseline_training.log"
    fpath = os.path.join(HERE, fname)

    with open(fpath, "r") as fb:

        for line in fb:

            # check time stamp
            x = re.search("^Current time ....*", line)
            if x is not None:
                x = re.search("\d+.\d+", line)
                data["training_time"].append(float(x.group()))

            # check epoch information
            x = re.search("^8498/8498.*", line)
            if x is not None:
                x = re.findall("-*\d+\.\d+", line)
                for m,v in zip(metrics,x):
                    data[m].append(float(v))

    out_fname = "baseline_training.json"
    out_fpath = os.path.join(HERE, out_fname)
    with open(out_fpath, "w") as fb:
        json.dump(data, fb)

if __name__ == "__main__":
    main()
