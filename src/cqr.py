import numpy as np


# Conformalized Quantile Regression, Romano, Patterson and Candes (2019).
# Everything here runs on saved predictions. The model is never touched.


def conformal_offset(scores, alpha):
    """The ceil((1-alpha)(n+1))-th smallest score.

    alpha <= 0 gives +inf, so the interval is the whole line.
    alpha >= 1 gives -inf, so the interval is empty.
    Both conventions are needed for the ACI guarantee to hold.
    """
    scores = np.asarray(scores).ravel()
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf

    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    if k < 1:
        return -np.inf

    return float(np.partition(scores, k - 1)[k - 1])


def scores_raw(y, lo, hi):
    """E = max(lo - y, y - hi). Negative inside the interval."""
    return np.maximum(lo - y, y - hi)


def scores_norm(y, lo, hi, scale):
    """Same score divided by the series scale, so series of different size
    can be pooled."""
    e = scores_raw(y, lo, hi)
    return e / scale[None, :, None]


def fit_pooled(y_cal, lo_cal, hi_cal, alpha):
    """B1: one offset for everything."""
    e = scores_raw(y_cal, lo_cal, hi_cal)
    return conformal_offset(e, alpha)


def apply_pooled(lo, hi, q):
    return lo - q, hi + q


def fit_horizon(y_cal, lo_cal, hi_cal, scale, alpha):
    """B2: one offset per horizon step, on scale-normalised scores."""
    e = scores_norm(y_cal, lo_cal, hi_cal, scale)
    h = e.shape[2]
    qs = np.zeros(h)
    for w in range(h):
        qs[w] = conformal_offset(e[:, :, w], alpha)
    return qs


def apply_horizon(lo, hi, qs, scale):
    """qs is (OUT_LEN,), scale is (n_series,)."""
    off = scale[None, :, None] * qs[None, None, :]
    return lo - off, hi + off


def fit_two_tail(y_cal, lo_cal, hi_cal, scale, alpha):
    """B3: a separate offset for each tail, each at level alpha/2."""
    e_lo = (lo_cal - y_cal) / scale[None, :, None]
    e_hi = (y_cal - hi_cal) / scale[None, :, None]
    h = e_lo.shape[2]
    q_lo = np.zeros(h)
    q_hi = np.zeros(h)
    for w in range(h):
        q_lo[w] = conformal_offset(e_lo[:, :, w], alpha / 2.0)
        q_hi[w] = conformal_offset(e_hi[:, :, w], alpha / 2.0)
    return q_lo, q_hi


def apply_two_tail(lo, hi, q_lo, q_hi, scale):
    off_lo = scale[None, :, None] * q_lo[None, None, :]
    off_hi = scale[None, :, None] * q_hi[None, None, :]
    return lo - off_lo, hi + off_hi
