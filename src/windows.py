import numpy as np

import config


# A window is a pair (series i, origin t0).
#   input  = y[i, t0-IN_LEN+1 : t0+1]
#   target = y[i, t0+1 : t0+1+OUT_LEN]
# So t0 is the last hour the model is allowed to see.


def split_origins(n_steps, protocol="daily"):
    """Return the origins of each split, counting back in days from the end.

    Returns a dict with keys train, val, calib, test.
    """
    out_len = config.OUT_LEN
    in_len = config.IN_LEN

    test_h = config.TEST_DAYS * 24
    calib_h = config.CALIB_DAYS * 24
    val_h = config.VAL_DAYS * 24

    # first target hour of each window
    test_start = n_steps - test_h
    calib_start = test_start - calib_h
    val_start = calib_start - val_h

    if protocol == "daily":
        step = 24
    elif protocol == "hourly":
        step = 1
    else:
        raise ValueError("protocol must be daily or hourly")

    def block(start, n_hours):
        # origins whose target lies fully inside [start, start+n_hours)
        o = []
        t = start
        while t + out_len <= start + n_hours:
            o.append(t - 1)
            t += step
        return np.array(o, dtype=np.int64)

    test = block(test_start, test_h)
    calib = block(calib_start, calib_h)
    val = block(val_start, val_h)

    # training origins: every hour whose whole target is before the val block
    first = in_len - 1
    last = val_start - out_len - 1
    if last < first:
        train = np.array([], dtype=np.int64)
    else:
        train = np.arange(first, last + 1, dtype=np.int64)

    return {"train": train, "val": val, "calib": calib, "test": test}


def check_splits(sp):
    """Make sure no target hour appears in two splits."""
    out_len = config.OUT_LEN
    seen = {}
    for name in ["train", "val", "calib", "test"]:
        hours = set()
        for t0 in sp[name]:
            for w in range(1, out_len + 1):
                hours.add(int(t0) + w)
        seen[name] = hours

    names = ["train", "val", "calib", "test"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = seen[names[i]] & seen[names[j]]
            if len(common) > 0:
                raise AssertionError(
                    "splits overlap: " + names[i] + " and " + names[j]
                )

    # order must be train < val < calib < test
    for i in range(len(names) - 1):
        a = seen[names[i]]
        b = seen[names[i + 1]]
        if len(a) > 0 and len(b) > 0:
            if max(a) >= min(b):
                raise AssertionError("splits out of order at " + names[i])
    return True


def calendar(n_steps):
    """Hour of day, day of week, month index for every time step.

    Step 0 is treated as a Monday midnight. The absolute date does not
    matter for the model, only the repeating pattern.
    """
    t = np.arange(n_steps)
    hour = t % 24
    dow = (t // 24) % 7
    month = ((t // 24) // 30) % 12
    return np.stack([hour, dow, month], axis=1).astype(np.int64)


def build_batch(y, cal, series_idx, origins):
    """Build one batch by plain numpy indexing, no python loop over windows.

    y          (n_series, n_steps)
    cal        (n_steps, 3)
    series_idx (B,)
    origins    (B,)

    Returns y_hist (B, IN_LEN), y_tgt (B, OUT_LEN),
            cal_hist (B, IN_LEN, 3), cal_fut (B, OUT_LEN, 3)
    """
    in_len = config.IN_LEN
    out_len = config.OUT_LEN

    series_idx = np.asarray(series_idx)
    origins = np.asarray(origins)

    hist_cols = origins[:, None] - in_len + 1 + np.arange(in_len)[None, :]
    fut_cols = origins[:, None] + 1 + np.arange(out_len)[None, :]

    y_hist = y[series_idx[:, None], hist_cols]
    y_tgt = y[series_idx[:, None], fut_cols]

    cal_hist = cal[hist_cols]
    cal_fut = cal[fut_cols]

    return y_hist, y_tgt, cal_hist, cal_fut


def insample(y):
    """Everything before the first test target hour, the in-sample part
    the paper's MASE denominator is taken over."""
    first_test = int(split_origins(y.shape[1], "daily")["test"][0]) + 1
    return y[:, :first_test]


def series_scale(y, origins_train):
    """One positive number per series, used to normalise conformity scores.

    Mean absolute value over the training part of the series, floored so a
    nearly all-zero series cannot blow the score up.
    """
    last = int(origins_train.max()) if len(origins_train) > 0 else y.shape[1] - 1
    s = np.abs(y[:, : last + 1]).mean(axis=1)
    floor = 0.1 * np.median(s)
    if floor <= 0:
        floor = 1e-6
    return np.maximum(s, floor)
