import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

import config
import metrics


# Phase 2. Does the nominal 80% interval actually cover 80%?
# This is the number the whole extension exists to fix, so it runs on its
# own and writes its own CSV.


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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    tag = sys.argv[2] if len(sys.argv) > 2 else "naive"
    protocol = sys.argv[3] if len(sys.argv) > 3 else "daily"

    preds, y = load(name, "test", protocol, tag)
    scale = np.load(os.path.join(config.ARRAYS, name + "_scale.npy"))

    lo = preds[:, :, :, 0]
    hi = preds[:, :, :, 2]
    target = 1.0 - config.ALPHA

    print("dataset", name, "model", tag, "protocol", protocol)
    print("")
    print("target coverage      ", round(target, 3))
    print("measured coverage    ", round(metrics.coverage(y, lo, hi), 4))
    print("normalised width     ", round(metrics.width(y, lo, hi), 4))
    print("crossing rate        ", round(metrics.crossing_rate(preds), 6))
    print("")

    rows = metrics.by_horizon(y, lo, hi, config.ALPHA)
    print("coverage by horizon")
    for r in rows:
        if r["horizon"] % 4 == 0 or r["horizon"] == 1:
            print("  h", r["horizon"], "->", round(r["coverage"], 4))

    # coverage split by series size, to justify the scale normalisation
    order = np.argsort(scale)
    n = len(order)
    print("")
    print("coverage by series size (quartiles, small to large)")
    for k in range(4):
        idx = order[k * n // 4 : (k + 1) * n // 4]
        print(
            "  q" + str(k + 1),
            round(metrics.coverage(y[:, idx], lo[:, idx], hi[:, idx]), 4),
        )

    df = pd.DataFrame(rows)
    df["dataset"] = name
    df["model"] = tag
    out = os.path.join(config.RESULTS, "diagnostic_" + name + "_" + tag + ".csv")
    df.to_csv(out, index=False)
    print("")
    print("wrote", out)


if __name__ == "__main__":
    main()
