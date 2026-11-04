"""Build each outer fold's features once, so experiments only refit classifiers.

This mirrors the feature half of `src.pipeline.fit_fold` and saves its matrices
instead of fitting the winner models. It lives outside `src/` so the selected
pipeline, its cache signature and the reference source hashes stay unchanged.

    python experiments/store.py --data data --output outputs/store --mode cv --jobs 2
"""

import argparse
import os
from pathlib import Path
import sys

# Same settings as run.py; feature construction is only reproducible under them.
SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
if __name__ == "__main__" and any(
    os.environ.get(name) != value for name, value in SETTINGS.items()
):
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **SETTINGS))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import features, state
from src.data import load_type_chart
from src.observations import describe, load, observation_features
from src.pipeline import prefix_subset, prepare, read_pickle, save_pickle, signature
from src.validation import N_FOLDS, SEEDS


def fold_path(store, seed, fold):
    return Path(store) / f"features_{seed}_{fold}.pkl.gz"


def build_fold(data, store, mode, seed, fold):
    """Return nothing; write A, Z, the additional blocks and the raw frame."""
    started = time.time()
    target = fold_path(store, seed, fold)
    key, _ = signature(data, mode)
    if target.exists():
        return f"{seed}/{fold}: cached"
    inputs = read_pickle(Path(store) / "inputs.pkl.gz")
    assert inputs["signature"] == key
    train = load(data / "train.jsonl")
    n = len(train)
    test_indices = np.flatnonzero(inputs["routing"][seed] == fold)
    if mode == "predict":
        all_test = load(data / "test.jsonl")
        test = [all_test[i] for i in test_indices]
        del all_test
    else:
        test = []
    battles = train + test
    views = [describe(b) for b in battles]
    tr, va = inputs["partitions"][seed][fold]
    query = np.concatenate([va, n + np.arange(len(test))])
    labels = np.concatenate([inputs["y"], np.full(len(test), -1)])
    groups = np.concatenate([inputs["groups"], inputs["test_groups"][test_indices]])
    assert not set(groups[tr]) & set(groups[query])
    context = dict(
        battles=battles,
        views=views,
        y=labels,
        groups=groups,
        chart=load_type_chart(data),
        static={
            "obs": features.numeric(
                [observation_features(b, v) for b, v in zip(battles, views)]
            )
        },
        resources=features.numeric(
            [state.resources(b, v) for b, v in zip(battles, views)]
        ),
    )
    A, Z = features.matrices(context, tr, query)
    original_indices = np.concatenate([np.arange(n), n + test_indices])
    prefixes = prefix_subset(inputs["prefixes"], original_indices)
    additional = features.additional(context, prefixes, tr, query, A, Z)
    raw_frame = inputs["raw"].iloc[original_indices].reset_index(drop=True)
    save_pickle(
        target,
        dict(
            signature=key,
            seed=seed,
            fold=fold,
            tr=tr,
            va=va,
            query=query,
            test_indices=test_indices,
            labels=labels,
            A=A,
            Z=Z,
            additional=additional,
            raw_train=raw_frame.iloc[tr].reset_index(drop=True),
            raw_query=raw_frame.iloc[query].reset_index(drop=True),
        ),
    )
    return f"{seed}/{fold}: built in {time.time() - started:.0f}s"


def load_fold(store, seed, fold):
    """Rebuild the five pipeline views from a cached fold.

    Returns a dict with `views` ({name: (X, Q)}), the outer indices and labels.
    Q holds validation rows first, then any routed test rows.
    """
    saved = read_pickle(fold_path(store, seed, fold))
    tr, query = saved["tr"], saved["query"]
    views = features.components(saved["A"], saved["Z"], tr, query, saved["additional"])
    X, Q = saved["raw_train"].copy(), saved["raw_query"].copy()
    for column in X:
        if column.startswith("c_"):
            levels = sorted(X[column].unique())
            X[column] = pd.Categorical(X[column], categories=levels)
            Q[column] = pd.Categorical(Q[column], categories=levels)
    views["raw_summary"] = (X, Q)
    labels = saved["labels"]
    return dict(
        views=views,
        tr=tr,
        va=saved["va"],
        test_indices=saved["test_indices"],
        y_train=labels[tr],
        y_valid=labels[saved["va"]],
        A=saved["A"],
        Z=saved["Z"],
        additional=saved["additional"],
        signature=saved["signature"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/store"))
    parser.add_argument("--mode", choices=["cv", "predict"], default="cv")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--only", type=int, nargs=2, metavar=("SEED", "FOLD"), help="build one fold"
    )
    args = parser.parse_args()
    data, store = args.data.resolve(), args.output.resolve()
    prepare(data, store, args.mode)
    tasks = [tuple(args.only)] if args.only else [
        (seed, fold) for seed in SEEDS for fold in range(N_FOLDS)
    ]
    # One task per child process, so each fold's memory is released on exit.
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=1) as pool:
        futures = [
            pool.submit(build_fold, data, store, args.mode, seed, fold)
            for seed, fold in tasks
        ]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
