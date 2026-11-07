import os

import numpy as np
import pandas as pd

import config


# LD2011_2014.txt from UCI. Put it in data/raw/.
# Separator is ';' and the decimal mark is ','.
# The first column is a timestamp, the rest are 370 customers.
# Row 1 is 2011-01-01 00:15, readings are every 15 minutes.
#
# 2011-2014 is 1461 days, so 35064 hours after summing each 4 readings.
# The paper keeps 2012-2014, which is the last 26304 of them.


def load(raw_file=None, out_file=None):
    if raw_file is None:
        raw_file = os.path.join(config.RAW, "LD2011_2014.txt")
    if out_file is None:
        out_file = os.path.join(config.PROCESSED, "electricity.npy")

    if not os.path.exists(raw_file):
        raise IOError("missing file: " + raw_file)

    df = pd.read_csv(raw_file, sep=";", decimal=",", index_col=0)
    print("raw table", df.shape, "(expected about (140256, 370))")

    x = df.values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)

    if x.shape[1] != 370:
        raise AssertionError(
            "expected 370 customers, got " + str(x.shape[1])
            + " - check the separator and that the timestamp column was"
            + " used as the index"
        )

    n_rows = x.shape[0]
    if n_rows % 4 != 0:
        print("dropping", n_rows % 4, "trailing rows so the hours divide evenly")
        x = x[: n_rows - (n_rows % 4)]

    n_hours = x.shape[0] // 4
    hourly = x.reshape(n_hours, 4, x.shape[1]).sum(axis=1)
    print("hours after aggregation", n_hours, "(expected 35064)")

    keep = config.get("electricity")["n_steps"]
    if n_hours < keep:
        raise AssertionError(
            "only " + str(n_hours) + " hours, need at least " + str(keep)
        )

    # keep the last 26304 hours, which is 2012-01-01 to 2014-12-31
    hourly = hourly[n_hours - keep :]

    y = hourly.T.astype(np.float32)

    print("electricity shape", y.shape)
    s = config.get("electricity")
    if y.shape != (s["n_series"], s["n_steps"]):
        raise AssertionError(
            "expected " + str((s["n_series"], s["n_steps"]))
            + " but got " + str(y.shape)
        )

    zero = float((y == 0).mean())
    print("share of exact zeros", round(zero, 4))
    print("(many series start late, so some zeros early on are normal)")

    np.save(out_file, y)
    return y


if __name__ == "__main__":
    load()
