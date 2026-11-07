import glob
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
import metrics
import windows
from losses import pinball_loss
from train import scale_window
from xseries import XTCN


# Traffic, baseline DeepTCN vs cross-series attention vs learned tau.
# One training step = one origin with all 963 lanes, so attention sees the
# whole panel.
#
#   python3 scripts/08_traffic_ext.py configs/traffic_*.yaml
#   python3 scripts/08_traffic_ext.py --table         just print the table

QS = config.QUANTILES
PAPER = {"QL50": 0.058, "QL90": 0.040, "NRMSE": 0.364, "SMAPE": 0.110, "MASE": 0.438}
COLS = ["QL50", "QL90", "NRMSE", "SMAPE", "MASE"]


def batch(y, cal, t0, dev):
    n = y.shape[0]
    si, oi = np.arange(n), np.full(n, t0)
    yh, yt, ch, cf = windows.build_batch(y, cal, si, oi)
    yh_s, yt_s, sc = scale_window(yh, yt)
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
    return T(yh_s), T(yt_s), T(si, torch.long), T(ch, torch.long), T(cf, torch.long), sc, yt


def predict(model, y, cal, origins, dev):
    model.eval()
    P, Y, loss = [], [], 0.0
    with torch.no_grad():
        for t0 in origins:
            yh, yt, si, ch, cf, sc, raw = batch(y, cal, t0, dev)
            p = model(yh, si, ch, cf)
            loss += float(pinball_loss(yt, p, QS))
            P.append(p.cpu().numpy() * sc[:, None, None])
            Y.append(raw)
    return np.stack(P), np.stack(Y), loss / len(origins)


def run(cfg, seed, y, cal, sp, mscale, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = XTCN(y.shape[0], config.IN_LEN, cfg["dilations"], cfg["channels"],
             cfg["emb_dim"], len(QS), attn=cfg["attn"], tau=cfg["tau"],
             s0=cfg["s0"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    best, state = float("inf"), None
    E = cfg["epochs"]
    for ep in range(E):
        if cfg["tau"]:
            m.s.fill_(cfg["s0"] * (cfg["s_end"] / cfg["s0"]) ** (ep / max(1, E - 1)))
        m.train()
        for t0 in rng.choice(sp["train"], cfg["steps_per_epoch"]):
            yh, yt, si, ch, cf, _, _ = batch(y, cal, t0, dev)
            loss = pinball_loss(yt, m(yh, si, ch, cf), QS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        _, _, v = predict(m, y, cal, sp["val"], dev)
        msg = "  ep %d val %.4f" % (ep, v)
        if cfg["attn"]:
            msg += "  alpha %.4f" % m.alpha.item()
        if cfg["tau"]:
            t = m.tau(torch.arange(y.shape[0], device=dev)).detach().cpu().numpy()
            msg += "  tau median %.1f  s %.1f" % (np.median(t), m.s.item())
        print(msg)
        if v < best:
            best = v
            state = {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)

    P, Y, _ = predict(m, y, cal, sp["test"], dev)
    r = metrics.point_metrics(Y.astype(np.float64), P.astype(np.float64), QS, mscale)
    r.update(row=cfg["name"], seed=seed, val=best)

    idx = torch.arange(y.shape[0], device=dev)
    extra = {}
    with torch.no_grad():
        if cfg["attn"]:
            r["alpha"] = m.alpha.item()
            extra["A"] = m.attention(idx).cpu().numpy()
            extra["E"] = m.E.weight.cpu().numpy()
        if cfg["tau"]:
            extra["tau"] = m.tau(idx).cpu().numpy()
    return r, extra


def coherence_test(A, E, y_train, n_null=200):
    """Is attention aimed at lanes that really move together? Attention-
    weighted mean correlation of deseasonalised lanes, against the same
    number with the rows of E shuffled (A relabelled by a permutation)."""
    n, T = y_train.shape
    T = T - T % 168
    r = y_train[:, :T].reshape(n, -1, 168)
    r = (r - r.mean(1, keepdims=True)).reshape(n, -1)
    C = np.corrcoef(r)
    np.fill_diagonal(C, 0)
    obs = float((A * C).sum() / n)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(n_null):
        p = rng.permutation(n)
        null.append(float((A[p][:, p] * C).sum() / n))
    null = np.array(null)
    return obs, null.mean(), null.std(), float((null >= obs).mean())


def figures(name, extra, y_train, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    base = os.path.join(config.RESULTS, "xseries_" + name)
    if "A" in extra:
        A, E = extra["A"], extra["E"]
        order = leaves_list(linkage(E, "average"))
        obs, mu, sd, p = coherence_test(A, E, y_train)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.log10(A[order][:, order] + 1e-6), cmap="viridis",
                       interpolation="nearest")
        fig.colorbar(im, label="log10 A")
        ax.set_title("A, lanes ordered by clustering E (alpha %.3f)\n"
                     "coherence %.3f vs shuffled-E %.3f +/- %.3f, p = %.3f"
                     % (alpha, obs, mu, sd, p), fontsize=9)
        fig.tight_layout()
        fig.savefig(base + "_A.png", dpi=110)
        plt.close(fig)
        print("  A coherence %.4f, shuffled-E null %.4f +/- %.4f, p = %.3f"
              % (obs, mu, sd, p))
    if "tau" in extra:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(extra["tau"], bins=40)
        ax.axvline(config.IN_LEN, color="k", ls="--", lw=1)
        t = "learned tau, 963 lanes (dashed = 168)"
        if alpha is not None:
            t += ", alpha %.3f" % alpha
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("tau (hours)")
        fig.tight_layout()
        fig.savefig(base + "_tau.png", dpi=110)
        plt.close(fig)
    print("  figures in results/xseries_" + name + "_*.png")


def table():
    files = sorted(glob.glob(os.path.join(config.RESULTS, "xseries_*.csv")))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    order = ["baseline", "attn", "tau", "both"]
    print("")
    print("traffic test, daily protocol, mean +/- std over seeds")
    print("row".ljust(10) + "".join(c.rjust(17) for c in COLS) + "   alpha")
    for row in order + sorted(set(df.row) - set(order)):
        d = df[df.row == row]
        if len(d) == 0:
            continue
        cells = "".join(("%.3f +/- %.3f" % (d[c].mean(), d[c].std(ddof=0))).rjust(17)
                        for c in COLS)
        a = ("%.3f" % d.alpha.mean()) if "alpha" in d and d.alpha.notna().any() else "-"
        print(row.ljust(10) + cells + "   " + a + "   (n=%d)" % len(d))
    print("paper".ljust(10) + "".join(("%.3f" % PAPER[c]).rjust(17) for c in COLS))


def main():
    if "--table" in sys.argv:
        table()
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.load(os.path.join(config.PROCESSED, "traffic.npy"))
    cal = windows.calendar(y.shape[1])[:, :2]
    sp = windows.split_origins(y.shape[1], "daily")
    mscale = metrics.mase_scale(windows.insample(y), config.MASE_PERIOD)
    y_train = y[:, : int(sp["train"].max()) + 1]

    for path in sys.argv[1:]:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
        rows = []
        for i, seed in enumerate(cfg["seeds"]):
            t0 = time.time()
            print("==", cfg["name"], "seed", seed, "on", dev)
            r, extra = run(cfg, seed, y, cal, sp, mscale, dev)
            print("  " + "  ".join("%s %.4f" % (c, r[c]) for c in COLS),
                  "(%.0fs)" % (time.time() - t0))
            rows.append(r)
            if i == 0 and extra:
                tag = cfg["name"] + "_seed" + str(seed)
                for k in ("A", "tau"):
                    if k in extra:
                        np.save(os.path.join(config.ARRAYS, "xseries_%s_%s.npy" % (k, tag)), extra[k])
                figures(cfg["name"], extra, y_train, r.get("alpha"))
        pd.DataFrame(rows).to_csv(
            os.path.join(config.RESULTS, "xseries_" + cfg["name"] + ".csv"), index=False
        )
    table()


if __name__ == "__main__":
    main()
