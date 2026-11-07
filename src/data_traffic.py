import os

import numpy as np

import config


# PEMS-SF from UCI. Put PEMS_train, PEMS_test and randperm in
# data/raw/PEMS-SF/.
#
# Each line of PEMS_train / PEMS_test is one day, written as
#   [v v v ...; v v v ...; ...]
# with 963 rows (sensors) and 144 columns (ten-minute readings).
# randperm says where each day belongs in calendar order.
#
# The direction of randperm is ambiguous: it could mean "day i belongs at
# position perm[i]" or "position i holds day perm[i]". Getting it backwards
# scrambles the calendar and quietly ruins every hour-of-day covariate, so
# day_continuity() below decides it from the data instead of guessing.


def parse_line(line):
    line = line.strip()
    if line.startswith("["):
        line = line[1:]
    if line.endswith("]"):
        line = line[:-1]
    rows = line.split(";")
    out = []
    for r in rows:
        vals = r.replace(",", " ").split()
        if len(vals) == 0:
            continue
        out.append([float(v) for v in vals])
    return np.array(out, dtype=np.float32)


def read_days(path):
    if not os.path.exists(path):
        raise IOError("missing file: " + path)
    days = []
    f = open(path, "r")
    for line in f:
        if len(line.strip()) == 0:
            continue
        days.append(parse_line(line))
    f.close()
    return days


def read_randperm(path):
    f = open(path, "r")
    txt = f.read()
    f.close()
    txt = txt.replace("[", " ").replace("]", " ").replace(",", " ")
    return [int(v) for v in txt.split()]


def day_continuity(days):
    """How much each day looks like the one after it.

    Traffic has a strong daily shape, so in the right calendar order
    neighbouring days correlate highly. In a scrambled order they do not.
    """
    prof = []
    for d in days:
        prof.append(d.mean(axis=0))
    prof = np.array(prof)

    c = []
    for i in range(len(prof) - 1):
        a = prof[i] - prof[i].mean()
        b = prof[i + 1] - prof[i + 1].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            c.append(float((a * b).sum() / den))
    return float(np.mean(c)) if len(c) > 0 else 0.0


def apply_perm(days, perm, direction):
    n = len(days)
    out = [None] * n
    for i in range(n):
        if direction == "forward":
            # day i belongs at position perm[i]
            out[perm[i] - 1] = days[i]
        else:
            # position i holds day perm[i]
            out[i] = days[perm[i] - 1]
    return out


def load(raw_dir=None, out_file=None):
    if raw_dir is None:
        raw_dir = os.path.join(config.RAW, "PEMS-SF")
    if out_file is None:
        out_file = os.path.join(config.PROCESSED, "traffic.npy")

    days = read_days(os.path.join(raw_dir, "PEMS_train"))
    days = days + read_days(os.path.join(raw_dir, "PEMS_test"))
    perm = read_randperm(os.path.join(raw_dir, "randperm"))

    print("read", len(days), "days")

    shapes = set()
    for d in days:
        shapes.add(d.shape)
    if len(shapes) != 1:
        raise AssertionError("days have different shapes: " + str(shapes))

    shape = list(shapes)[0]
    print("each day is", shape, "(expected (963, 144))")
    if shape[1] != 144:
        raise AssertionError(
            "expected 144 ten-minute readings per day, got " + str(shape[1])
            + " - the parser is reading the file wrongly"
        )

    if len(days) != len(perm):
        raise AssertionError(
            "got " + str(len(days)) + " days but " + str(len(perm))
            + " randperm entries"
        )

    # decide the direction of randperm from the data
    fwd = apply_perm(days, perm, "forward")
    bwd = apply_perm(days, perm, "backward")
    c_raw = day_continuity(days)
    c_fwd = day_continuity(fwd)
    c_bwd = day_continuity(bwd)

    print("")
    print("day-to-day continuity (higher means the order is right)")
    print("  file order     ", round(c_raw, 4))
    print("  perm forward   ", round(c_fwd, 4))
    print("  perm backward  ", round(c_bwd, 4))

    if c_fwd >= c_bwd:
        ordered = fwd
        print("  -> using forward")
    else:
        ordered = bwd
        print("  -> using backward")

    if max(c_fwd, c_bwd) < c_raw:
        print("  WARNING: neither direction beats the raw file order.")
        print("  Check the randperm file before trusting the result.")

    # sum every 6 ten-minute readings to get hours
    hourly = []
    for d in ordered:
        d = np.nan_to_num(d, nan=0.0)
        h = d.reshape(d.shape[0], 24, 6).sum(axis=2)
        hourly.append(h)

    y = np.concatenate(hourly, axis=1).astype(np.float32)

    print("")
    print("traffic shape", y.shape)
    s = config.get("traffic")
    if y.shape[0] != s["n_series"]:
        raise AssertionError(
            "expected " + str(s["n_series"]) + " series, got " + str(y.shape[0])
        )
    if y.shape[1] != s["n_steps"]:
        print(
            "NOTE: expected", s["n_steps"], "hours but got", y.shape[1],
            "- set n_steps in config.py to", y.shape[1],
        )

    print("value range", round(float(y.min()), 3), "to", round(float(y.max()), 3))
    print("(sums of 6 readings, so up to about 6, not 1)")

    np.save(out_file, y)
    return y


if __name__ == "__main__":
    load()
