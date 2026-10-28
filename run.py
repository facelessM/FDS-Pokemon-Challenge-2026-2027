"""Reproduce the selected model from the competition files."""

import argparse
import os
from pathlib import Path
import sys

# Set these before Python creates hash tables or numerical libraries load.
SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
if any(os.environ.get(name) != value for name, value in SETTINGS.items()):
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **SETTINGS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cv", "predict", "collect"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--worker", type=int, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = args.data.resolve()
    output = (args.output or Path("outputs") / args.mode).resolve()
    from src.pipeline import collect, fit_fold, prepare, run_folds

    if args.worker:
        fit_fold(data, output, args.mode, *args.worker)
        return
    if args.mode == "collect":
        from src.pipeline import read_pickle

        mode = read_pickle(output / "inputs.pkl.gz")["provenance"]["mode"]
        collect(data, output, mode)
        return
    prepare(data, output, args.mode)
    run_folds(data, output, args.mode, args.jobs)
    collect(data, output, args.mode)


if __name__ == "__main__":
    main()
