import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

import config
import metrics
import windows


# The final scores, the way the paper reports them (section 4.2): 24 hours
# ahead for each of the last seven days, one model trained before the first
# window, QL50/QL90 (Table 4) and NRMSE/SMAPE/MASE on the median (Table 5).
# That is the daily protocol; hourly is ours, not the paper's.
#
#   python3 scripts/06_paper_metrics.py traffic
#   python3 scripts/06_paper_metrics.py traffic daily seed1 negbin_seed1
#
# With no tags it scores every model whose test arrays are in arrays/.
# These are the raw model outputs, as in the paper. Calibration moves only
# the interval ends, never the median, so NRMSE/SMAPE/MASE do not change.

# paper Tables 4 and 5, "-" where the paper gives no number
PAPER = {
    "electricity": [
        ("ETS", 0.100, 0.050, 0.838, 0.156, 1.247),
        ("auto.arima", 0.142, 0.054, None, None, None),
        ("DeepAR-t", 0.068, 0.033, 0.550, 0.110, 0.931),
        ("SQF-RNN", 0.066, 0.035, 0.518, 0.113, 0.937),
        ("DeepState", 0.043, 0.025, None, None, None),
        ("DeepTCN-Quantile", 0.057, 0.029, 0.523, 0.118, 0.926),
        ("DeepTCN-Gaussian", 0.062, 0.039, None, None, None),
    ],
    "traffic": [
        ("ETS", 0.360, 0.325, 0.872, 0.594, 1.881),
        ("auto.arima", 0.246, 0.140, None, None, None),
        ("DeepAR-t", 0.117, 0.090, 0.396, 0.104, 0.442),
        ("SQF-RNN", 0.119, 0.090, 0.381, 0.117, 0.449),
        ("DeepState", 0.084, 0.058, None, None, None),
        ("DeepTCN-Quantile", 0.058, 0.040, 0.364, 0.110, 0.438),
        ("DeepTCN-Gaussian", 0.071, 0.049, None, None, None),
    ],
}

COLS = ["QL50", "QL90", "NRMSE", "SMAPE", "MASE"]


def find_tags(name, protocol):
    pat = os.path.join(
        config.ARRAYS, "preds_" + name + "_test_" + protocol + "_*.npy"
    )
    pre = "preds_" + name + "_test_" + protocol + "_"
    return sorted(os.path.basename(p)[len(pre) : -4] for p in glob.glob(pat))


def label(tag):
    """seed1 -> quantile, negbin_seed1 -> negbin, naive -> naive"""
    if tag.startswith("seed"):
        return "quantile"
    if "_seed" in tag:
        return tag.split("_seed")[0]
    return tag


def fmt(v):
    return "-" if v is None else "%.3f" % v


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    protocol = sys.argv[2] if len(sys.argv) > 2 else "daily"
    tags = sys.argv[3:] if len(sys.argv) > 3 else find_tags(name, protocol)
    if len(tags) == 0:
        raise IOError("no test arrays for " + name + " " + protocol + " in arrays/")

    y_all = np.load(os.path.join(config.PROCESSED, name + ".npy"))
    mscale = metrics.mase_scale(windows.insample(y_all), config.MASE_PERIOD)

    rows = []
    for tag in tags:
        stem = name + "_test_" + protocol + "_" + tag + ".npy"
        preds = np.load(os.path.join(config.ARRAYS, "preds_" + stem))
        y = np.load(os.path.join(config.ARRAYS, "actuals_" + stem))
        r = metrics.point_metrics(
            y.astype(np.float64), preds.astype(np.float64),
            config.QUANTILES, mscale,
        )
        r["model"] = "DeepTCN-" + label(tag) if tag != "naive" else "seasonal naive"
        r["tag"] = tag
        r["dataset"] = name
        r["protocol"] = protocol
        r["n_origins"] = y.shape[0]
        rows.append(r)

    print("dataset", name, "protocol", protocol,
          "MASE period", config.MASE_PERIOD)
    print("")
    head = "model".ljust(24) + "".join(c.rjust(8) for c in COLS)
    print("ours")
    print(head)
    for r in rows:
        print(r["model"].ljust(24) + "".join(fmt(r[c]).rjust(8) for c in COLS))

    if name in PAPER:
        print("")
        print("paper, Tables 4 and 5")
        print(head)
        for p in PAPER[name]:
            print(p[0].ljust(24) + "".join(fmt(v).rjust(8) for v in p[1:]))
        if protocol != "daily":
            print("(the paper's numbers are for the daily protocol)")

    out = os.path.join(
        config.RESULTS, "paper_metrics_" + name + "_" + protocol + ".csv"
    )
    pd.DataFrame(rows).to_csv(out, index=False)
    print("")
    print("wrote", out)


if __name__ == "__main__":
    main()
