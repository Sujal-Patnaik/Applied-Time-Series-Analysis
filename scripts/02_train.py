import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import train


def main():
    # python3 scripts/02_train.py traffic 1 [--dist gaussian|negbin|zip] [--resume]
    args   = [a for a in sys.argv[1:] if not a.startswith("--")]
    dist   = sys.argv[sys.argv.index("--dist") + 1] if "--dist" in sys.argv else "quantile"
    if dist in args:
        args.remove(dist)
    name   = args[0] if len(args) > 0 else "synthetic"
    seed   = int(args[1]) if len(args) > 1 else 1
    resume = "--resume" in sys.argv
    train.train(name, seed, resume=resume, dist=dist)


if __name__ == "__main__":
    main()
