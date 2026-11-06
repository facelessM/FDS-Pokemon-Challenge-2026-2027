"""Reproduce the selected model from the competition files."""

import argparse
import os
from pathlib import Path
import sys

# Set these before Python creates hash tables or numerical libraries load.
SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
# The feature corrections are read at import time by src/options.py.
SETTINGS["FDS_FEATURE_FIXES"] = "1" if "--feature-fixes" in sys.argv else "0"
if any(os.environ.get(name) != value for name, value in SETTINGS.items()):
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **SETTINGS))


def main():
    """Parse the arguments and run one mode.

    cv and predict build inputs, fit all fifteen folds and collect them;
    collect rebuilds the outputs from finished folds; stage fits the second
    stage on a finished predict run. --worker fits a single fold.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cv", "predict", "collect", "stage"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument(
        "--feature-fixes",
        action="store_true",
        help="use the corrected features (see src/options.py)",
    )
    parser.add_argument(
        "--view",
        choices=["blend", "tactical"],
        default="blend",
        help="stage: base model",
    )
    parser.add_argument("--signals", default="team", help="stage: team or team,spikes")
    parser.add_argument("--name", help="stage: submission file name")
    parser.add_argument("--worker", type=int, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = args.data.resolve()
    output = (args.output or Path("outputs") / args.mode).resolve()
    from src.pipeline import collect, fit_fold, prepare, run_folds

    if args.worker:
        fit_fold(data, output, args.mode, *args.worker)
        return
    if args.mode == "stage":
        from src.second_stage import run

        output = (args.output or Path("outputs") / "predict").resolve()
        run(
            data,
            output,
            args.view,
            args.signals,
            args.name or f"{args.view}_{args.signals}",
        )
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
