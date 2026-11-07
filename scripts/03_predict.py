import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    # python3 scripts/03_predict.py traffic deeptcn daily 1 [--dist negbin]
    # writes arrays tagged seed1 for the quantile head, negbin_seed1 etc.
    # otherwise; pass that tag to 04_diagnostic.py and 05_calibrate.py
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dist = sys.argv[sys.argv.index("--dist") + 1] if "--dist" in sys.argv else "quantile"
    if dist in args:
        args.remove(dist)
    name = args[0] if len(args) > 0 else "synthetic"
    model = args[1] if len(args) > 1 else "deeptcn"
    protocol = args[2] if len(args) > 2 else "daily"
    seed = int(args[3]) if len(args) > 3 else 1

    if model == "naive":
        import baseline_naive

        baseline_naive.run(name, protocol, seed)
    else:
        import predict

        predict.run(name, seed, protocol=protocol, dist=dist)


if __name__ == "__main__":
    main()
