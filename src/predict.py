import os

import numpy as np
import torch

import config
import distributions
import windows
from train import build_model, make_batch, run_tag


def predict_split(model, y, cal, origins, n_series, device, log_target,
                  dist="quantile", count_unit=1.0):
    """Returns preds (n_origins, n_series, OUT_LEN, n_q) and
    actuals (n_origins, n_series, OUT_LEN), both on the original scale.

    A parametric head is turned into the same quantiles, so everything
    downstream treats every head alike."""
    n_q = len(config.QUANTILES)
    out_len = config.OUT_LEN

    preds = np.zeros((len(origins), n_series, out_len, n_q), dtype=np.float32)
    acts = np.zeros((len(origins), n_series, out_len), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for k in range(len(origins)):
            t0 = origins[k]
            si = np.arange(n_series)
            oi = np.full(n_series, t0)

            y_hist, y_tgt, _, _ = windows.build_batch(y, cal, si, oi)
            acts[k] = y_tgt

            yh, _, sidx, ch, cf, sc = make_batch(
                y, cal, si, oi, device, log_target, dist, count_unit
            )
            p = model(yh, sidx, ch, cf).cpu().numpy()

            if dist != "quantile":
                p = distributions.quantiles(
                    dist, p, sc.cpu().numpy(), config.QUANTILES
                )
                if distributions.is_count(dist):
                    p = p / count_unit
                elif log_target:
                    p = np.expm1(p)
                preds[k] = p
                continue

            # undo the window scaling, then the log if there was one
            h = np.log1p(y_hist) if log_target else y_hist
            s = np.maximum(np.abs(h).mean(axis=1), 1e-6)
            p = p * s[:, None, None]

            if log_target:
                p = np.expm1(p)

            preds[k] = p

    return preds, acts


def run(name, seed=1, device=None, protocol="daily", dist="quantile"):
    distributions.check(dist)
    s = config.get(name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y = np.load(os.path.join(config.PROCESSED, name + ".npy"))
    n_series = y.shape[0]
    n_steps = y.shape[1]

    cal = windows.calendar(n_steps)
    sp = windows.split_origins(n_steps, protocol)
    sp_daily = windows.split_origins(n_steps, "daily")

    model = build_model(name, n_series, dist).to(device)
    ckpt = os.path.join(config.ARRAYS, name + "_" + run_tag(dist, seed) + ".pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    scale = windows.series_scale(y, sp_daily["train"])
    np.save(os.path.join(config.ARRAYS, name + "_scale.npy"), scale)

    for split in ["val", "calib", "test"]:
        preds, acts = predict_split(
            model, y, cal, sp[split], n_series, device, s["log_target"],
            dist, s.get("count_unit", 1.0),
        )
        tag = name + "_" + split + "_" + protocol + "_" + run_tag(dist, seed)
        np.save(os.path.join(config.ARRAYS, "preds_" + tag + ".npy"), preds)
        np.save(os.path.join(config.ARRAYS, "actuals_" + tag + ".npy"), acts)
        print(split, preds.shape)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    protocol = sys.argv[3] if len(sys.argv) > 3 else "daily"
    dist = sys.argv[sys.argv.index("--dist") + 1] if "--dist" in sys.argv else "quantile"
    run(name, seed, protocol=protocol, dist=dist)
