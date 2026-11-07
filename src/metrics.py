import numpy as np


# Shapes used everywhere:
#   preds   (n_origins, n_series, OUT_LEN, n_quantiles)
#   actuals (n_origins, n_series, OUT_LEN)
#   lo, hi  (n_origins, n_series, OUT_LEN)


def pinball(y, yhat, q):
    d = y - yhat
    return np.maximum(q * d, (q - 1.0) * d)


def quantile_loss(y, yhat, q):
    """QL as in the paper: sum of pinball over sum of |y|, no factor 2."""
    num = pinball(y, yhat, q).sum()
    den = np.abs(y).sum()
    if den == 0:
        return 0.0
    return float(num / den)


def smape(y, yhat):
    num = np.abs(2.0 * (y - yhat))
    den = np.abs(y) + np.abs(yhat)
    r = np.zeros_like(num, dtype=np.float64)
    ok = den > 0
    r[ok] = num[ok] / den[ok]
    return float(r.mean())


def nrmse(y, yhat):
    rmse = np.sqrt(((y - yhat) ** 2).mean())
    den = np.abs(y).mean()
    if den == 0:
        return 0.0
    return float(rmse / den)


def mase_scale(y_train, period=24):
    """One scale per series: mean |y[t] - y[t-period]| over the in-sample
    part, the MASE denominator in the paper. Not the same thing as the
    mean |y| scale used to normalise conformity scores.

    y_train is (n_series, n_steps).
    """
    d = np.abs(y_train[:, period:] - y_train[:, :-period])
    s = d.mean(axis=1)
    floor = 1e-6
    return np.maximum(s, floor)


def mase(y, yhat, scale):
    """scale is (n_series,), broadcast over origins and horizon."""
    err = np.abs(y - yhat)
    sc = scale[None, :, None]
    return float((err / sc).mean())


def coverage(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean())


def width(y, lo, hi):
    """Mean interval width over mean |y|, finite intervals only."""
    w = hi - lo
    ok = np.isfinite(w)
    if ok.sum() == 0:
        return float("nan")
    den = np.abs(y).mean()
    if den == 0:
        return float("nan")
    return float(w[ok].mean() / den)


def winkler(y, lo, hi, alpha):
    w = hi - lo
    below = y < lo
    above = y > hi
    score = w.astype(np.float64).copy()
    score[below] += (2.0 / alpha) * (lo[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - hi[above])
    ok = np.isfinite(score)
    if ok.sum() == 0:
        return float("nan")
    den = np.abs(y).mean()
    if den == 0:
        return float("nan")
    return float(score[ok].mean() / den)


def crossing_rate(preds):
    """Share of forecasts where any adjacent quantile pair is out of order."""
    bad = np.zeros(preds.shape[:3], dtype=bool)
    for j in range(preds.shape[3] - 1):
        bad = bad | (preds[:, :, :, j] > preds[:, :, :, j + 1])
    return float(bad.mean())


def infinite_share(lo, hi):
    bad = (~np.isfinite(lo)) | (~np.isfinite(hi))
    return float(bad.mean())


def by_horizon(y, lo, hi, alpha):
    """Coverage, width and Winkler for each horizon step."""
    h = y.shape[2]
    rows = []
    for w in range(h):
        yy = y[:, :, w]
        ll = lo[:, :, w]
        hh = hi[:, :, w]
        rows.append(
            {
                "horizon": w + 1,
                "coverage": coverage(yy, ll, hh),
                "width": width(yy, ll, hh),
                "winkler": winkler(yy, ll, hh, alpha),
            }
        )
    return rows


def by_day(y, lo, hi, alpha, origins_per_day=1):
    """Coverage for each block of origins, used for the test week plot."""
    n = y.shape[0]
    rows = []
    d = 0
    i = 0
    while i < n:
        j = min(i + origins_per_day, n)
        yy = y[i:j]
        ll = lo[i:j]
        hh = hi[i:j]
        rows.append(
            {
                "day": d + 1,
                "coverage": coverage(yy, ll, hh),
                "width": width(yy, ll, hh),
                "winkler": winkler(yy, ll, hh, alpha),
            }
        )
        i = j
        d += 1
    return rows


def point_metrics(y, preds, quantiles, mscale):
    """The paper's accuracy numbers (Tables 4 and 5): QL for every quantile,
    then SMAPE, NRMSE and MASE on the median. mscale is from mase_scale."""
    med = quantiles.index(0.5)
    yhat = preds[:, :, :, med]

    out = {}
    for j in range(len(quantiles)):
        q = quantiles[j]
        out["QL" + str(int(q * 100))] = quantile_loss(y, preds[:, :, :, j], q)

    out["SMAPE"] = smape(y, yhat)
    out["NRMSE"] = nrmse(y, yhat)
    out["MASE"] = mase(y, yhat, mscale)
    return out


def headline(y, preds, lo, hi, quantiles, mscale, alpha):
    """Every number for one model on one dataset, as a flat dict.
    mscale is the MASE denominator from mase_scale."""
    out = point_metrics(y, preds, quantiles, mscale)
    out["coverage"] = coverage(y, lo, hi)
    out["width"] = width(y, lo, hi)
    out["winkler"] = winkler(y, lo, hi, alpha)
    out["crossing"] = crossing_rate(preds)
    out["inf_share"] = infinite_share(lo, hi)
    return out
