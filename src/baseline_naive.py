import os

import numpy as np

import config
import windows


# Seasonal naive with empirical quantiles.
#
#   point forecast for t0 + w  =  y[t0 + w - 24]   (same hour, previous day)
#   quantiles                  =  point + the empirical quantile of the
#                                 training residual at that horizon
#
# Two reasons this is here. It is an honest extra baseline for the report,
# and it needs no GPU, so the calibration code can be written and tested
# before DeepTCN has finished training.


def point_forecast(y, origins, out_len):
    """(n_origins, n_series, out_len) using the same hour one day earlier."""
    n_series = y.shape[0]
    out = np.zeros((len(origins), n_series, out_len), dtype=np.float32)
    for k in range(len(origins)):
        t0 = int(origins[k])
        for w in range(1, out_len + 1):
            out[k, :, w - 1] = y[:, t0 + w - 24]
    return out


def actuals(y, origins, out_len):
    n_series = y.shape[0]
    out = np.zeros((len(origins), n_series, out_len), dtype=np.float32)
    for k in range(len(origins)):
        t0 = int(origins[k])
        out[k] = y[:, t0 + 1 : t0 + 1 + out_len]
    return out


def fit_spread(y, origins_train, out_len, quantiles, n_sample=400, seed=0):
    """Residual quantiles per horizon step, learned on training origins only.

    The spread is in units of the series scale, so one table works for every
    series.
    """
    rng = np.random.default_rng(seed)
    if len(origins_train) > n_sample:
        picks = rng.choice(origins_train, size=n_sample, replace=False)
    else:
        picks = origins_train

    p = point_forecast(y, picks, out_len)
    a = actuals(y, picks, out_len)

    scale = windows.series_scale(y, origins_train)
    res = (a - p) / scale[None, :, None]

    spread = np.zeros((out_len, len(quantiles)))
    for w in range(out_len):
        for j in range(len(quantiles)):
            spread[w, j] = np.quantile(res[:, :, w].ravel(), quantiles[j])
    return spread, scale


def forecast(y, origins, spread, scale, out_len, quantiles):
    p = point_forecast(y, origins, out_len)
    n_q = len(quantiles)
    out = np.zeros(p.shape + (n_q,), dtype=np.float32)
    for j in range(n_q):
        out[:, :, :, j] = p + scale[None, :, None] * spread[None, None, :, j]
    return np.maximum(out, 0.0)


def run(name, protocol="daily", seed=1):
    y = np.load(os.path.join(config.PROCESSED, name + ".npy"))
    n_steps = y.shape[1]
    out_len = config.OUT_LEN
    qs = config.QUANTILES

    sp = windows.split_origins(n_steps, protocol)
    sp_daily = windows.split_origins(n_steps, "daily")
    windows.check_splits(sp_daily)

    spread, scale = fit_spread(y, sp_daily["train"], out_len, qs, seed=seed)
    np.save(os.path.join(config.ARRAYS, name + "_scale.npy"), scale)

    for split in ["val", "calib", "test"]:
        o = sp[split]
        preds = forecast(y, o, spread, scale, out_len, qs)
        acts = actuals(y, o, out_len)
        tag = name + "_" + split + "_" + protocol + "_naive"
        np.save(os.path.join(config.ARRAYS, "preds_" + tag + ".npy"), preds)
        np.save(os.path.join(config.ARRAYS, "actuals_" + tag + ".npy"), acts)
        print(split, preds.shape)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    protocol = sys.argv[2] if len(sys.argv) > 2 else "daily"
    run(name, protocol)
