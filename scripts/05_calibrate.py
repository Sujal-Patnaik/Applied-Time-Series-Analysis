import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

import aci
import config
import cqr
import metrics
import windows


# Phases 3 and 4. Reads saved predictions, writes the result CSVs that the
# R scripts plot. Nothing here touches a model.


def load(name, split, protocol, tag):
    p = os.path.join(
        config.ARRAYS,
        "preds_" + name + "_" + split + "_" + protocol + "_" + tag + ".npy",
    )
    a = os.path.join(
        config.ARRAYS,
        "actuals_" + name + "_" + split + "_" + protocol + "_" + tag + ".npy",
    )
    return np.load(p), np.load(a)


def bounds(preds):
    return preds[:, :, :, 0].astype(np.float64), preds[:, :, :, 2].astype(np.float64)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    tag = sys.argv[2] if len(sys.argv) > 2 else "naive"
    protocol = sys.argv[3] if len(sys.argv) > 3 else "daily"

    alpha = config.ALPHA
    qs = config.QUANTILES

    scale = np.load(os.path.join(config.ARRAYS, name + "_scale.npy"))

    # MASE needs the in-sample series, not the conformal scale above
    y_all = np.load(os.path.join(config.PROCESSED, name + ".npy"))
    mscale = metrics.mase_scale(windows.insample(y_all), config.MASE_PERIOD)

    p_cal, y_cal = load(name, "calib", protocol, tag)
    p_test, y_test = load(name, "test", protocol, tag)
    p_val, y_val = load(name, "val", protocol, tag)

    lo_cal, hi_cal = bounds(p_cal)
    lo_test, hi_test = bounds(p_test)
    lo_val, hi_val = bounds(p_val)

    y_cal = y_cal.astype(np.float64)
    y_test = y_test.astype(np.float64)
    y_val = y_val.astype(np.float64)

    models = {}

    # ---- baseline, no calibration
    models["base"] = (lo_test, hi_test)

    # ---- B1, one pooled offset
    q1 = cqr.fit_pooled(y_cal, lo_cal, hi_cal, alpha)
    models["cqr_pooled"] = cqr.apply_pooled(lo_test, hi_test, q1)
    print("B1 pooled offset", round(float(q1), 4))

    # ---- B2, one offset per horizon on scale-normalised scores
    q2 = cqr.fit_horizon(y_cal, lo_cal, hi_cal, scale, alpha)
    models["cqr_horizon"] = cqr.apply_horizon(lo_test, hi_test, q2, scale)
    print("B2 offsets h1", round(q2[0], 4), "h24", round(q2[-1], 4))

    # ---- B3, the two tails separately
    q3lo, q3hi = cqr.fit_two_tail(y_cal, lo_cal, hi_cal, scale, alpha)
    models["cqr_twotail"] = cqr.apply_two_tail(
        lo_test, hi_test, q3lo, q3hi, scale
    )

    # ---- ACI, with gamma chosen on the validation split
    delay = aci.make_delay(config.OUT_LEN, protocol)
    cal_scores = cqr.scores_norm(y_cal, lo_cal, hi_cal, scale)

    half = len(y_val) // 2
    if half >= 1:
        v_cal_scores = cqr.scores_norm(
            y_val[:half], lo_val[:half], hi_val[:half], scale
        )
        gamma, score = aci.pick_gamma(
            y_val[half:],
            lo_val[half:],
            hi_val[half:],
            v_cal_scores,
            scale,
            alpha,
            delay,
            [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            metrics.winkler,
        )
    else:
        gamma = 0.05
        score = float("nan")
    print("gamma chosen on validation:", gamma)

    lo_a, hi_a, atrace, ebar = aci.run_aci(
        y_test, lo_test, hi_test, cal_scores, scale, alpha, gamma, delay
    )
    models["aci"] = (lo_a, hi_a)

    bound = aci.coverage_bound(alpha, gamma, delay, len(y_test))
    print("ACI bound value", round(bound, 3), "(above 1 means it says nothing)")

    # ---- results
    head_rows = []
    hor_rows = []
    day_rows = []

    per_day = 1 if protocol == "daily" else 24

    for m in models:
        lo, hi = models[m]
        h = metrics.headline(y_test, p_test, lo, hi, qs, mscale, alpha)
        h["dataset"] = name
        h["model"] = m
        h["base"] = tag
        h["protocol"] = protocol
        h["gamma"] = gamma if m == "aci" else ""
        head_rows.append(h)

        for r in metrics.by_horizon(y_test, lo, hi, alpha):
            r["dataset"] = name
            r["model"] = m
            r["protocol"] = protocol
            hor_rows.append(r)

        for r in metrics.by_day(y_test, lo, hi, alpha, per_day):
            r["dataset"] = name
            r["model"] = m
            r["protocol"] = protocol
            day_rows.append(r)

    suffix = name + "_" + tag + "_" + protocol

    pd.DataFrame(head_rows).to_csv(
        os.path.join(config.RESULTS, "headline_" + suffix + ".csv"), index=False
    )
    pd.DataFrame(hor_rows).to_csv(
        os.path.join(config.RESULTS, "coverage_by_horizon_" + suffix + ".csv"),
        index=False,
    )
    pd.DataFrame(day_rows).to_csv(
        os.path.join(config.RESULTS, "coverage_by_day_" + suffix + ".csv"),
        index=False,
    )

    at = []
    for t in range(atrace.shape[0]):
        for w in range(atrace.shape[1]):
            at.append(
                {
                    "dataset": name,
                    "protocol": protocol,
                    "origin": t + 1,
                    "horizon": w + 1,
                    "alpha": atrace[t, w],
                    "miss_rate": ebar[t, w],
                }
            )
    pd.DataFrame(at).to_csv(
        os.path.join(config.RESULTS, "alpha_trace_" + suffix + ".csv"), index=False
    )

    # one series drawn in full, for the intuition figure
    i = int(np.argmax(scale))
    ex = []
    lo_h, hi_h = models["cqr_horizon"]
    for t in range(y_test.shape[0]):
        for w in range(y_test.shape[2]):
            ex.append(
                {
                    "step": t * y_test.shape[2] + w + 1,
                    "actual": y_test[t, i, w],
                    "lo_base": lo_test[t, i, w],
                    "hi_base": hi_test[t, i, w],
                    "lo_cqr": lo_h[t, i, w],
                    "hi_cqr": hi_h[t, i, w],
                }
            )
    pd.DataFrame(ex).to_csv(
        os.path.join(config.RESULTS, "example_series_" + suffix + ".csv"), index=False
    )

    print("")
    print("model          coverage   width   winkler")
    for h in head_rows:
        print(
            h["model"].ljust(14),
            str(round(h["coverage"], 4)).ljust(10),
            str(round(h["width"], 4)).ljust(7),
            round(h["winkler"], 4),
        )
    print("")
    print("wrote 5 csv files to results/ with suffix", suffix)


if __name__ == "__main__":
    main()
