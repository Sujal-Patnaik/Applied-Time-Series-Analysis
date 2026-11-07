import os

import numpy as np
import torch

import config
import distributions
import windows
from losses import pinball_loss
from model import DeepTCN


def run_tag(dist, seed):
    """Suffix for checkpoints and prediction arrays. The quantile model keeps
    the old plain seedN so existing files still line up."""
    if dist == "quantile":
        return "seed" + str(seed)
    return dist + "_seed" + str(seed)


def build_model(name, n_series, dist):
    s = config.get(name)
    if dist == "quantile":
        return DeepTCN(
            n_series, s["dilations"], s["channels"], s["kernel"],
            len(config.QUANTILES),
        )
    return DeepTCN(
        n_series, s["dilations"], s["channels"], s["kernel"], 2, positive=False
    )


def scale_window(y_hist, y_tgt):
    """Divide both by the mean of the input window, so series of different
    size look alike to the model. Returns the scale so it can be undone."""
    s = np.abs(y_hist).mean(axis=1)
    s = np.maximum(s, 1e-6)
    return y_hist / s[:, None], y_tgt / s[:, None], s


def to_fit_scale(y_hist, y_tgt, log_target, dist, count_unit):
    """Raw windows -> what the parametric head is fitted on, plus nu.

    The network sees y_hist / nu, the target stays unscaled; see
    distributions.py for why it is 1 + mean.
    """
    if distributions.is_count(dist):
        y_hist = np.round(y_hist * count_unit)
        y_tgt = np.round(y_tgt * count_unit)
    elif log_target:
        y_hist = np.log1p(y_hist)
        y_tgt = np.log1p(y_tgt)
    nu = 1.0 + np.abs(y_hist).mean(axis=1)
    return y_hist / nu[:, None], y_tgt, nu


def make_batch(y, cal, series_idx, origins, device, log_target,
               dist="quantile", count_unit=1.0):
    """Returns y_hist, y_tgt, series_idx, cal_hist, cal_fut, scale as tensors.
    scale is only used by the parametric heads."""
    y_hist, y_tgt, cal_hist, cal_fut = windows.build_batch(
        y, cal, series_idx, origins
    )

    if dist == "quantile":
        if log_target:
            y_hist = np.log1p(y_hist)
            y_tgt = np.log1p(y_tgt)
        y_hist, y_tgt, sc = scale_window(y_hist, y_tgt)
    else:
        y_hist, y_tgt, sc = to_fit_scale(
            y_hist, y_tgt, log_target, dist, count_unit
        )

    return (
        torch.tensor(y_hist, dtype=torch.float32, device=device),
        torch.tensor(y_tgt, dtype=torch.float32, device=device),
        torch.tensor(series_idx, dtype=torch.long, device=device),
        torch.tensor(cal_hist, dtype=torch.long, device=device),
        torch.tensor(cal_fut, dtype=torch.long, device=device),
        torch.tensor(sc, dtype=torch.float32, device=device),
    )


def batch_loss(dist, yt, out, sc):
    if dist == "quantile":
        return pinball_loss(yt, out, config.QUANTILES)
    return distributions.nll(dist, yt, out, sc)


def sample_windows(n_series, origins, n, rng):
    si = rng.integers(0, n_series, size=n)
    oi = origins[rng.integers(0, len(origins), size=n)]
    return si, oi


def val_loss(model, y, cal, origins, n_series, device, log_target,
             dist="quantile", count_unit=1.0):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for t0 in origins:
            si = np.arange(n_series)
            oi = np.full(n_series, t0)
            yh, yt, s, ch, cf, sc = make_batch(
                y, cal, si, oi, device, log_target, dist, count_unit
            )
            pred = model(yh, s, ch, cf)
            total += float(batch_loss(dist, yt, pred, sc))
            count += 1
    model.train()
    if count == 0:
        return float("inf")
    return total / count


def train(name, seed=1, device=None, out_dir=None, n_per_epoch=20000,
          resume=False, dist="quantile"):
    distributions.check(dist)
    s = config.get(name)
    count_unit = s.get("count_unit", 1.0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if out_dir is None:
        out_dir = config.ARRAYS

    y = np.load(os.path.join(config.PROCESSED, name + ".npy"))
    n_series = y.shape[0]
    n_steps = y.shape[1]

    cal = windows.calendar(n_steps)
    sp = windows.split_origins(n_steps, "daily")
    windows.check_splits(sp)

    print("train origins", len(sp["train"]))
    print("val origins", len(sp["val"]))
    print("calib origins", len(sp["calib"]))
    print("test origins", len(sp["test"]))
    print("output head", dist)

    model = build_model(name, n_series, dist).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])

    # paths for both checkpoints
    tag         = run_tag(dist, seed)
    ckpt        = os.path.join(out_dir, name + "_" + tag + ".pt")
    resume_ckpt = os.path.join(out_dir, name + "_" + tag + "_resume.pt")

    best       = float("inf")
    best_epoch = -1
    bad        = 0
    start_epoch = 0

    if resume and os.path.exists(resume_ckpt):
        # restore everything so training continues exactly where it left off
        state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best        = state["best"]
        best_epoch  = state["best_epoch"]
        bad         = state["bad"]
        rng         = np.random.default_rng(seed)  # recreate rng ...
        rng.bit_generator.state = state["rng"]     # ... then restore its state
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed from epoch {state['epoch']}  "
              f"best so far {round(best, 5)} at epoch {best_epoch}")
    else:
        # fresh run — seed everything from scratch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if resume:
            print("no resume checkpoint found, starting fresh")

    n_batches = max(1, n_per_epoch // s["batch_size"])

    for epoch in range(start_epoch, s["epochs"]):
        run = 0.0
        for b in range(n_batches):
            si, oi = sample_windows(
                n_series, sp["train"], s["batch_size"], rng
            )
            yh, yt, sidx, ch, cf, sc = make_batch(
                y, cal, si, oi, device, s["log_target"], dist, count_unit
            )

            pred = model(yh, sidx, ch, cf)
            loss = batch_loss(dist, yt, pred, sc)

            if not torch.isfinite(loss):
                raise RuntimeError("loss went to nan at epoch " + str(epoch))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run += loss.item()

        v = val_loss(
            model,
            y,
            cal,
            sp["val"],
            n_series,
            device,
            s["log_target"],
            dist,
            count_unit,
        )
        print("epoch", epoch, "train", round(run / n_batches, 5), "val", round(v, 5))

        if v < best:
            best = v
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), ckpt)  # best-model checkpoint
        else:
            bad += 1
            if bad >= s["patience"]:
                print("early stop at epoch", epoch)
                break

        # save full resume checkpoint after every epoch
        torch.save(
            {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  opt.state_dict(),
                "best":       best,
                "best_epoch": best_epoch,
                "bad":        bad,
                "rng":        rng.bit_generator.state,
                "torch_rng":  torch.get_rng_state(),
            },
            resume_ckpt,
        )

    print("best epoch", best_epoch, "val", round(best, 5))
    return ckpt


if __name__ == "__main__":
    import sys

    name   = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    seed   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    resume = "--resume" in sys.argv
    dist   = sys.argv[sys.argv.index("--dist") + 1] if "--dist" in sys.argv else "quantile"
    train(name, seed, resume=resume, dist=dist)
