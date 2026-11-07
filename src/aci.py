import numpy as np

from cqr import conformal_offset


# Adaptive Conformal Inference, Gibbs and Candes (2021).
#
# One level per horizon step, updated with the miss rate averaged over
# series. A forecast made at origin t for horizon w is only verifiable at
# t + w, so origin t may only use errors from origins at or before t - delay[w].


def make_delay(out_len, protocol):
    if protocol == "daily":
        # a whole day is revealed before the next daily origin
        return np.ones(out_len, dtype=np.int64)
    elif protocol == "hourly":
        return np.arange(1, out_len + 1, dtype=np.int64)
    else:
        raise ValueError("protocol must be daily or hourly")


def run_aci(y, lo, hi, cal_scores, scale, alpha, gamma, delay):
    """Walk the test origins in time order, adapting the level as we go.

    y, lo, hi   (n_origins, n_series, OUT_LEN)  base model interval
    cal_scores  (n_cal, n_series, OUT_LEN)      normalised calibration scores
    scale       (n_series,)
    delay       (OUT_LEN,)

    Returns lo_new, hi_new, alpha_trace, ebar.
    """
    n_orig = y.shape[0]
    h = y.shape[2]

    lo_new = np.zeros_like(lo, dtype=np.float64)
    hi_new = np.zeros_like(hi, dtype=np.float64)
    alpha_trace = np.zeros((n_orig, h))
    ebar = np.zeros((n_orig, h))

    # pre-sort the calibration scores once per horizon
    cal_by_h = []
    for w in range(h):
        s = cal_scores[:, :, w].ravel()
        cal_by_h.append(s[np.isfinite(s)])

    for t in range(n_orig):
        for w in range(h):
            r = t - int(delay[w])

            if r >= 0:
                a = alpha + gamma * np.sum(alpha - ebar[: r + 1, w])
            else:
                a = alpha

            alpha_trace[t, w] = a

            q = conformal_offset(cal_by_h[w], a)

            if np.isinf(q) and q > 0:
                l = np.full(y.shape[1], -np.inf)
                u = np.full(y.shape[1], np.inf)
            elif np.isinf(q) and q < 0:
                # empty set: an interval nothing can fall into
                l = np.full(y.shape[1], np.inf)
                u = np.full(y.shape[1], -np.inf)
            else:
                off = scale * q
                l = lo[t, :, w] - off
                u = hi[t, :, w] + off

            lo_new[t, :, w] = l
            hi_new[t, :, w] = u

            miss = (y[t, :, w] < l) | (y[t, :, w] > u)
            ebar[t, w] = miss.mean()

    return lo_new, hi_new, alpha_trace, ebar


def coverage_bound(alpha_1, gamma, delay, n_orig):
    """The right hand side of the guarantee, for the worst horizon.

    Bigger than 1 means the bound says nothing, which is what happens with
    only 7 daily origins.
    """
    d = float(np.max(delay))
    top = max(alpha_1, 1.0 - alpha_1) + d * gamma
    return float(top / (n_orig * gamma))


def pick_gamma(y, lo, hi, cal_scores, scale, alpha, delay, grid, winkler_fn):
    """Choose gamma on the validation split by interval score.

    winkler_fn(y, lo, hi, alpha) -> float, passed in so this file does not
    need to import metrics.
    """
    best = None
    best_g = grid[0]
    for g in grid:
        l, u, _, _ = run_aci(y, lo, hi, cal_scores, scale, alpha, g, delay)
        s = winkler_fn(y, l, u, alpha)
        if not np.isfinite(s):
            continue
        if best is None or s < best:
            best = s
            best_g = g
    return best_g, best
