import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import config
import windows


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"

    if name == "traffic":
        import data_traffic

        y = data_traffic.load()
    elif name == "electricity":
        import data_electricity

        y = data_electricity.load()
    elif name == "synthetic":
        import data_synthetic

        y = data_synthetic.load()
    else:
        raise ValueError("unknown dataset " + name)

    n_steps = y.shape[1]
    sp = windows.split_origins(n_steps, "daily")
    windows.check_splits(sp)

    print("")
    print("dataset", name, "shape", y.shape)
    for k in ["train", "val", "calib", "test"]:
        o = sp[k]
        if len(o) == 0:
            print(k, "0 origins")
        else:
            print(
                k,
                len(o),
                "origins, target hours",
                int(o[0]) + 1,
                "to",
                int(o[-1]) + config.OUT_LEN,
            )

    sph = windows.split_origins(n_steps, "hourly")
    print("hourly test origins", len(sph["test"]))
    print("splits ok")


if __name__ == "__main__":
    main()
