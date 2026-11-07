import numpy as np


# Synthetic panel with a known memory length per series, for checking that
# the learned tau means something before any real data is touched.
#
#   A  period-7 seasonal + noise            needs ~7 steps
#   B  period-24 seasonal + noise           needs ~24 steps
#   C  AR(1), phi = 0.3                     needs ~1-2 steps
#   D  type A, but only the last 3 history  needs ~3 steps, the rest is
#      values are real, the rest is 0       padding
#
# Each series has its own random seasonal profile, and window origins are
# random, so the phase can only be read off the history: the series
# embedding cannot memorise it. No calendar covariates are given for the
# same reason.
#
# Windows are cut from one long series per id, in time order: train first,
# then val, then test, so val and test are held out in time.

TYPES = ["A", "B", "C", "D"]


def make_series(kind, n_steps, rng, noise):
    t = np.arange(n_steps)
    if kind in ("A", "D", "B"):
        p = 24 if kind == "B" else 7
        prof = rng.normal(0.0, 1.0, size=p)
        prof = (prof - prof.mean()) / (prof.std() + 1e-8)
        return prof[t % p] + noise * rng.normal(0.0, 1.0, size=n_steps)
    if kind == "C":
        e = rng.normal(0.0, 1.0, size=n_steps)
        x = np.zeros(n_steps)
        for i in range(1, n_steps):
            x[i] = 0.3 * x[i - 1] + e[i]
        return x
    raise ValueError("unknown type " + kind)


def panel(n_per_type=500, in_len=64, out_len=12, n_train=6, n_val=2,
          n_test=2, noise=0.3, d_keep=3, seed=0):
    """Returns a dict split -> (x (W, in_len), y (W, out_len), sid (W,), kind (W,))
    and the type of every series id."""
    rng = np.random.default_rng(seed)
    win = in_len + out_len
    n_win = n_train + n_val + n_test
    # one block of 2 * win per window, origin drawn inside it, so windows
    # of the same series never overlap and the phase is random
    block = 2 * win
    n_steps = n_win * block

    out = {k: ([], [], [], []) for k in ["train", "val", "test"]}
    series_type = []
    sid = 0
    for kind in TYPES:
        for _ in range(n_per_type):
            y = make_series(kind, n_steps, rng, noise)
            for w in range(n_win):
                start = w * block + rng.integers(0, block - win + 1)
                x = y[start : start + in_len].copy()
                f = y[start + in_len : start + win]
                if kind == "D":
                    x[: in_len - d_keep] = 0.0
                split = (
                    "train" if w < n_train
                    else "val" if w < n_train + n_val
                    else "test"
                )
                for lst, v in zip(out[split], [x, f, sid, kind]):
                    lst.append(v)
            series_type.append(kind)
            sid += 1

    res = {}
    for k, (xs, ys, ss, ks) in out.items():
        res[k] = (
            np.array(xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(ss, dtype=np.int64),
            np.array(ks),
        )
    return res, np.array(series_type)
