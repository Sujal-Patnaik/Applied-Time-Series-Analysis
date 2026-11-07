import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import torch

import config
import metrics
import predict
import train
import windows


# Extensions on a 320-lane subset of traffic, sized to run in ~10-15 min on
# an 8-core CPU (two processes x 4 threads).
#
#   python3 scripts/09_traffic_subset.py ext2     quantile / negbin / zip, main pipeline
#   python3 scripts/09_traffic_subset.py ext34    baseline / attn / tau / both (xseries)
#   python3 scripts/09_traffic_subset.py table    print both result tables
#
# Subset: all 440 days (so splits, test week and MASE are the paper's),
# 320 of 963 lanes, 40 from each of 8 bands of mean training occupancy.

NAME = "traffic_sub"
N_LANES = 320
N_BANDS = 8
N_PER_EPOCH = 8000          # ext2: windows per epoch (batch 32 -> 250 steps)
XS_EPOCHS, XS_STEPS = 9, 100  # ext34: 900 steps, one step = all 320 lanes
COLS = ["QL50", "QL90", "NRMSE", "SMAPE", "MASE"]
PAPER = {"QL50": 0.058, "QL90": 0.040, "NRMSE": 0.364, "SMAPE": 0.110, "MASE": 0.438}


def make_subset():
    out = os.path.join(config.PROCESSED, NAME + ".npy")
    lanes_f = os.path.join(config.ARRAYS, NAME + "_lanes.npy")
    if os.path.exists(out) and os.path.exists(lanes_f):
        return np.load(out), np.load(lanes_f)
    y = np.load(os.path.join(config.PROCESSED, "traffic.npy"))
    sp = windows.split_origins(y.shape[1], "daily")
    m = y[:, : int(sp["train"].max()) + 1].mean(axis=1)
    order = np.argsort(m)
    rng = np.random.default_rng(0)
    per = N_LANES // N_BANDS
    lanes = np.concatenate([
        rng.choice(band, per, replace=False) for band in np.array_split(order, N_BANDS)
    ])
    lanes = np.sort(lanes)   # keep the original lane order (neighbours stay together)
    sub = y[lanes].astype(np.float32)
    np.save(out, sub)
    np.save(lanes_f, lanes)
    print("subset", sub.shape, "mean occupancy", round(float(m[lanes].mean()), 4),
          "vs full", round(float(m.mean()), 4))
    return sub, lanes


def mase_scale(y):
    return metrics.mase_scale(windows.insample(y), config.MASE_PERIOD)


def score(name_tag, y_mscale):
    stem = NAME + "_test_daily_" + name_tag + ".npy"
    p = np.load(os.path.join(config.ARRAYS, "preds_" + stem)).astype(np.float64)
    a = np.load(os.path.join(config.ARRAYS, "actuals_" + stem)).astype(np.float64)
    return metrics.point_metrics(a, p, config.QUANTILES, y_mscale)


def ext2():
    torch.set_num_threads(4)
    y, _ = make_subset()
    ms = mase_scale(y)
    rows = []
    for dist in ["quantile", "negbin", "zip"]:
        t0 = time.time()
        print("== ext2", dist, flush=True)
        train.train(NAME, 1, n_per_epoch=N_PER_EPOCH, dist=dist)
        predict.run(NAME, 1, protocol="daily", dist=dist)
        r = score(train.run_tag(dist, 1), ms)
        r.update(model="DeepTCN-" + dist, minutes=round((time.time() - t0) / 60, 2))
        rows.append(r)
        print("  " + "  ".join("%s %.4f" % (c, r[c]) for c in COLS), "(%.1f min)" % r["minutes"], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS, "subset_ext2.csv"), index=False)


def ext34():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "tx", os.path.join(ROOT, "scripts", "08_traffic_ext.py"))
    tx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tx)

    y, _ = make_subset()
    cal = windows.calendar(y.shape[1])[:, :2]
    sp = windows.split_origins(y.shape[1], "daily")
    ms = mase_scale(y)
    y_train = y[:, : int(sp["train"].max()) + 1]
    base = dict(dilations=[1, 2, 4, 8, 16, 20, 32, 32, 32], channels=16, emb_dim=32,
                lr=0.01, epochs=XS_EPOCHS, steps_per_epoch=XS_STEPS, s0=16.0, s_end=1.0)
    rows = []
    for name, attn, tau in [("baseline", False, False), ("attn", True, False),
                            ("tau", False, True), ("both", True, True)]:
        t0 = time.time()
        print("== ext34", name, flush=True)
        cfg = dict(base, name=name, attn=attn, tau=tau)
        r, extra = tx.run(cfg, 1, y, cal, sp, ms, "cpu")
        if "A" in extra:
            obs, mu, sd, p = tx.coherence_test(extra["A"], extra["E"], y_train)
            r.update(coherence=obs, null_mean=mu, null_sd=sd, p_value=p)
        if "tau" in extra:
            t = extra["tau"]
            r.update(tau_median=float(np.median(t)), tau_min=float(t.min()), tau_max=float(t.max()))
        r["minutes"] = round((time.time() - t0) / 60, 2)
        rows.append(r)
        print("  " + "  ".join("%s %.4f" % (c, r[c]) for c in COLS), "(%.1f min)" % r["minutes"], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS, "subset_ext34.csv"), index=False)


def table():
    print("traffic subset: %d lanes, all 440 days, last 7 days, daily protocol, seed 1" % N_LANES)
    for f, label in [("subset_ext2.csv", "Ext 2: output heads (main DeepTCN)"),
                     ("subset_ext34.csv", "Ext 3+4: memory tau / cross-series attention (xseries)")]:
        path = os.path.join(config.RESULTS, f)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        key = "model" if "model" in d else "row"
        print("")
        print(label)
        print("  %-18s" % "model" + "".join("%9s" % c for c in COLS) + "   min")
        for _, r in d.iterrows():
            print("  %-18s" % r[key] + "".join("%9.4f" % r[c] for c in COLS) + "  %4.1f" % r["minutes"])
        if "alpha" in d:
            for _, r in d[d.alpha.notna()].iterrows():
                print("  %s: alpha %.4f, coherence %.4f vs shuffled %.4f +/- %.4f (p = %.3f)" % (
                    r[key], r.alpha, r.coherence, r.null_mean, r.null_sd, r.p_value))
        if "tau_median" in d:
            for _, r in d[d.tau_median.notna()].iterrows():
                print("  %s: tau median %.1f h (range %.1f-%.1f)" % (r[key], r.tau_median, r.tau_min, r.tau_max))
    print("")
    print("  %-18s" % "paper (963 lanes)" + "".join("%9.4f" % PAPER[c] for c in COLS))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "table"
    {"ext2": ext2, "ext34": ext34, "table": table, "subset": make_subset}[part]()
