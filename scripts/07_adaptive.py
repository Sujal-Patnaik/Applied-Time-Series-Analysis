import copy
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import torch
import yaml

import config
import data_panel
import metrics
from adaptive import GatedTCN
from losses import pinball_loss


# Adaptive receptive field on the synthetic panel: trains every ablation row
# in the YAML file, checks the tau gate, writes the tables and the plot.
#
#   python3 scripts/07_adaptive.py configs/adaptive_synthetic.yaml
#   python3 scripts/07_adaptive.py configs/adaptive_synthetic.yaml --rows global full
#
# Rows (gate:)
#   none     baseline, the whole window                      (control)
#   fixed    hard cutoff at fixed_len steps
#   global   one learned tau for all series                  (key ablation)
#   series   tau per window from n_observed + zero ratio     (the idea)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def no_cal(b, n):
    return torch.zeros((b, n, 3), dtype=torch.long)


def predict(model, x, sid, out_len, bs=2048):
    model.eval()
    preds, taus = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb, sb = to_t(x[i : i + bs]), to_t(sid[i : i + bs], torch.long)
            p = model(xb, sb, no_cal(len(xb), xb.shape[1]), no_cal(len(xb), out_len))
            preds.append(p.numpy())
            t = model.last_tau
            taus.append(np.full(len(xb), np.nan) if t is None else t.numpy())
    return np.concatenate(preds), np.concatenate(taus)


def val_score(model, data, qs, out_len):
    x, y, sid, _ = data["val"]
    p, _ = predict(model, x, sid, out_len)
    return float(
        pinball_loss(to_t(y), to_t(p), qs)
    )


def train_one(cfg, data, n_series, gate, seed, fixed_len=None, log=None):
    m, tr, pc = cfg["model"], cfg["train"], cfg["panel"]
    qs = cfg["quantiles"]
    in_len, out_len = pc["in_len"], pc["out_len"]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedTCN(
        gate, n_series, m["dilations"], m["channels"], m["kernel"], len(qs),
        in_len, positive=False, fixed_len=fixed_len,
    )
    # the gate gets its own, smaller learning rate and is frozen for the
    # first gate_warmup epochs: an untrained encoder finds old history to be
    # pure noise and would shut the gate before it learns to use it
    gate_p = list(model.gate.parameters())
    gate_ids = {id(p) for p in gate_p}
    rest = [p for p in model.parameters() if id(p) not in gate_ids]
    groups = [{"params": rest, "lr": tr["lr"]}]
    if gate_p:
        groups.append({"params": gate_p, "lr": tr["lr"] * tr.get("gate_lr_mult", 1.0)})
    opt = torch.optim.Adam(groups)
    warm = tr.get("gate_warmup", 0)

    x, y, sid, _ = data["train"]
    n = len(x)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(tr["epochs"]):
        frac = (ep - warm) / max(1, tr["epochs"] - 1 - warm)
        model.gate.anneal(frac, tr["s_end"])
        for p in gate_p:
            p.requires_grad_(ep >= warm)
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, tr["batch_size"]):
            j = order[i : i + tr["batch_size"]]
            if len(j) < 2:
                continue
            xb, yb = to_t(x[j]), to_t(y[j])
            sb = to_t(sid[j], torch.long)
            p = model(xb, sb, no_cal(len(j), in_len), no_cal(len(j), out_len))
            loss = pinball_loss(yb, p, qs)
            if ep >= warm:
                loss = loss + model.penalty(tr["lambda"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        v = val_score(model, data, qs, out_len)
        if log:
            msg = "    ep %2d  val %.4f  s %.2f" % (ep, v, float(model.gate.s))
            if model.gate.learned:
                _, tv = predict(model, data["val"][0], data["val"][2], out_len)
                kv = data["val"][3]
                msg += "  tau " + " ".join(
                    "%s %.1f" % (k, np.median(tv[kv == k])) for k in data_panel.TYPES
                )
            log(msg)
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tr["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best


def scores(y, p, kinds, qs):
    """QL50/QL90 overall and per series type."""
    out = []
    for k in ["all"] + data_panel.TYPES:
        sel = np.ones(len(y), bool) if k == "all" else kinds == k
        r = {"type": k}
        for j, q in enumerate(qs):
            if q in (0.5, 0.9):
                r["QL" + str(int(q * 100))] = metrics.quantile_loss(
                    y[sel], p[sel, :, j], q
                )
        out.append(r)
    return out


def gate_check(tau_df):
    med = tau_df.groupby("type")["tau"].median()
    ok = (med["C"] < med["A"] < med["B"]) and (med["D"] < med[["A", "B", "C"]].min())
    return ok, med


def plot_tau(tau_df, true_scale, in_len, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, no plot (the csv has the data)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    order = ["D", "C", "A", "B"]
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        t = tau_df.loc[tau_df.type == k, "tau"].values
        ax[0].scatter(i + rng.uniform(-0.2, 0.2, len(t)), t, s=4, alpha=0.4)
        ax[0].plot([i - 0.3, i + 0.3], [true_scale[k]] * 2, "k--", lw=1)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels([k + " (" + str(true_scale[k]) + ")" for k in order])
    ax[0].set_ylabel("learned tau (steps)")
    ax[0].set_title("tau by series type, dashed = true memory")
    ax[0].axhline(in_len, color="grey", lw=0.8)

    ax[1].scatter(tau_df["n_observed"], tau_df["tau"], c=tau_df["type"].map(
        {k: i for i, k in enumerate(order)}), s=4, cmap="viridis")
    ax[1].set_xlabel("observed steps in window")
    ax[1].set_ylabel("learned tau (steps)")
    ax[1].set_title("tau vs history length")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: 07_adaptive.py config.yaml [--rows name ...]")
    cfg = load_cfg(sys.argv[1])
    rows = cfg["rows"]
    if "--rows" in sys.argv:
        keep = sys.argv[sys.argv.index("--rows") + 1 :]
        rows = [r for r in rows if r["name"] in keep]

    torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
    qs = cfg["quantiles"]
    pc = cfg["panel"]
    tag = cfg["tag"]
    data, series_type = data_panel.panel(**pc)
    n_series = len(series_type)
    xt, yt, st, kt = data["test"]
    print("panel", {k: v[0].shape for k, v in data.items()}, "series", n_series)

    res, tau_rows, gate_rows = [], [], []
    for seed in cfg["seeds"]:
        for row in rows:
            name, gate = row["name"], row["gate"]
            t0 = time.time()
            print("== seed", seed, "row", name, "gate", gate)

            mdl, v = train_one(
                cfg, data, n_series, gate, seed,
                row.get("fixed_len"), log=print if cfg.get("verbose") else None,
            )
            p, tau = predict(mdl, xt, st, pc["out_len"])
            extra = {}
            if not np.isnan(tau).all():
                nobs = (np.cumsum(xt != 0, axis=1) > 0).sum(axis=1)
                df = pd.DataFrame({"series": st, "type": kt, "tau": tau,
                                   "n_observed": nobs})
                df = df.groupby(["series", "type"], as_index=False).mean()
                df["row"], df["seed"] = name, seed
                tau_rows.append(df)
                ok, med = gate_check(df)
                gate_rows.append(dict(row=name, seed=seed, passed=ok,
                                      **{"median_tau_" + k: med[k] for k in med.index}))
                print("   median tau by type", med.round(2).to_dict(),
                      "-> gate", "PASSED" if ok else "FAILED")

            for r in scores(yt, p, kt, qs):
                r.update(row=name, seed=seed, **extra)
                res.append(r)
            print("   QL50 all %.4f  (%.0fs)" % (res[-5]["QL50"], time.time() - t0))

    os.makedirs(config.RESULTS, exist_ok=True)
    base = os.path.join(config.RESULTS, "adaptive_" + tag)
    df = pd.DataFrame(res)
    df.to_csv(base + "_rows.csv", index=False)

    summ = df.groupby(["row", "type"])[["QL50", "QL90"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(base + "_summary.csv", index=False)

    print("")
    print("test QL, mean +/- std over seeds", cfg["seeds"])
    piv = summ[summ.type == "all"].set_index("row")
    order = [r["name"] for r in rows]
    for name in order:
        if name in piv.index:
            r = piv.loc[name]
            print("  %-22s QL50 %.4f +/- %.4f   QL90 %.4f +/- %.4f" % (
                name, r.QL50_mean, np.nan_to_num(r.QL50_std),
                r.QL90_mean, np.nan_to_num(r.QL90_std)))

    if tau_rows:
        tdf = pd.concat(tau_rows)
        tdf.to_csv(base + "_tau.csv", index=False)
        gdf = pd.DataFrame(gate_rows)
        gdf.to_csv(base + "_gate.csv", index=False)
        print("")
        print("tau gate (C < A < B, D smallest)")
        print(gdf.round(2).to_string(index=False))
        main_row = cfg.get("plot_row", "per_series")
        sel = tdf[(tdf.row == main_row) & (tdf.seed == cfg["seeds"][0])]
        if len(sel):
            plot_tau(sel, cfg["true_scale"], pc["in_len"], base + "_tau.png")


if __name__ == "__main__":
    main()
