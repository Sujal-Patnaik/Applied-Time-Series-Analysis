import os

ROOT = os.path.dirname(os.path.abspath(__file__))

RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
ARRAYS = os.path.join(ROOT, "arrays")
RESULTS = os.path.join(ROOT, "results")

# window sizes, same for every dataset
IN_LEN = 168
OUT_LEN = 24

QUANTILES = [0.1, 0.5, 0.9]

# split sizes, counted back in whole days from the end of the series
TEST_DAYS = 7
CALIB_DAYS = 7
VAL_DAYS = 14

# output head: "quantile" (non-parametric, pinball loss) or one of the
# parametric heads in src/distributions.py, fitted by likelihood
DISTS = ["quantile", "gaussian", "negbin", "zip"]

# seasonal period m in the MASE denominator, mean |y[t] - y[t-m]|
# over the in-sample part (paper section 4.2.1); hourly data, so a day
MASE_PERIOD = 24

# nominal miscoverage, so the interval is [q0.1, q0.9]
ALPHA = 0.2

# one entry per dataset
SETTINGS = {
    "traffic": {
        "n_series": 963,
        "n_steps": 10560,
        "dilations": [1, 2, 4, 8, 16, 32],
        "channels": 16,
        "kernel": 2,
        "batch_size": 32,
        "lr": 0.005,
        "epochs": 100,
        "patience": 5,   # was 10
        "log_target": False,
        # negbin/zip fit round(y * count_unit); occupancy to 0.01
        "count_unit": 100.0,
    },
    "electricity": {
        "n_series": 370,
        "n_steps": 26304,
        "dilations": [1, 2, 4, 8, 16, 32],
        "channels": 32,
        "kernel": 2,
        "batch_size": 64,
        "lr": 0.005,
        "epochs": 60,    # raised to allow resume beyond epoch 25
        "patience": 5,   # was 10
        "log_target": True,   # quantile and gaussian only
        "count_unit": 1.0,    # whole kW
    },
    # 320-lane stratified subset of traffic, full 440 days (scripts/09_traffic_subset.py);
    # same model settings as traffic, shorter training budget
    "traffic_sub": {
        "n_series": 320,
        "n_steps": 10560,
        "dilations": [1, 2, 4, 8, 16, 32],
        "channels": 16,
        "kernel": 2,
        "batch_size": 32,
        "lr": 0.005,
        "epochs": 30,
        "patience": 5,
        "log_target": False,
        "count_unit": 100.0,
    },
    "synthetic": {
        "n_series": 40,
        "n_steps": 2400,
        "dilations": [1, 2, 4, 8],
        "channels": 16,
        "kernel": 2,
        "batch_size": 64,
        "lr": 0.005,
        "epochs": 3,
        "patience": 10,
        "log_target": False,
        "count_unit": 10.0,
    },
}


def get(name):
    if name not in SETTINGS:
        raise ValueError("unknown dataset " + name)
    return SETTINGS[name]
