import os

import numpy as np

import config


# Fake hourly data with a daily and weekly shape plus noise.
# Used to test the whole pipeline before the real datasets are ready.
#
# Two things are built in on purpose, because real data has both and the
# two extensions fix different ones:
#
#   1. the noise grows slowly over the whole series, so a model fitted on
#      the training part is too confident everywhere later on  -> CQR fixes
#   2. the last 7 days are noisier still, a shift the calibration week
#      never saw                                               -> ACI fixes
#
# Without (1) CQR has nothing to do, which is not what the real datasets
# look like.


def load(name="synthetic", out_file=None, seed=0):
    if out_file is None:
        out_file = os.path.join(config.PROCESSED, name + ".npy")

    s = config.get(name)
    n = s["n_series"]
    t = s["n_steps"]

    rng = np.random.default_rng(seed)

    hour = np.arange(t) % 24
    dow = (np.arange(t) // 24) % 7

    # noise level rises from 0.10 to about 0.26 across the series
    ramp = 0.10 * (1.0 + 1.6 * np.arange(t) / float(t))

    y = np.zeros((n, t), dtype=np.float32)
    for i in range(n):
        level = 5.0 + 20.0 * rng.random()
        daily = 1.0 + 0.6 * np.sin(2 * np.pi * hour / 24.0 + rng.random())
        weekly = 1.0 - 0.25 * (dow >= 5)
        noise = rng.normal(0.0, 1.0, size=t) * ramp
        y[i] = level * daily * weekly * (1.0 + noise)

    # an extra jump in the last 7 days only
    tail = 7 * 24
    y[:, t - tail :] *= 1.0 + rng.normal(0.0, 0.12, size=(n, tail))

    y = np.maximum(y, 0.0).astype(np.float32)

    print("synthetic shape", y.shape)
    np.save(out_file, y)
    return y


if __name__ == "__main__":
    load()
