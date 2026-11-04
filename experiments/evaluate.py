"""Refit winner classifiers on cached folds and compare their OOF predictions.

Run `experiments/store.py` first. Then, from the repository root:

    python experiments/evaluate.py reproduce --store outputs/store

or, in a notebook or script:

    from experiments.evaluate import evaluate, compare, pipeline_fit
    new = evaluate("outputs/store", my_fit, name="my_idea")
    compare(new, "reference")

Each run saves its per-component probabilities in `<store>/experiments/<name>.npz`,
so a later blend can reuse components without refitting them.
"""

import argparse
import json
import os
from pathlib import Path
import sys

SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
if __name__ == "__main__" and any(
    os.environ.get(name) != value for name, value in SETTINGS.items()
):
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **SETTINGS))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from experiments.store import load_fold
from src.models import lgbm_final, xgb_final
from src.pipeline import array_digest, read_pickle
from src.validation import COMPONENTS, N_FOLDS, SEEDS, THRESHOLD


def pipeline_fit(name, X, y, Q):
    """The selected method: LightGBM for raw rows, LightGBM + XGBoost otherwise."""
    factories = [lgbm_final] if name == "raw_summary" else [lgbm_final, xgb_final]
    return np.mean([f().fit(X, y).predict_proba(Q)[:, 1] for f in factories], axis=0)


def _fit_fold(store, seed, fold, fit, components, transform):
    fold_data = load_fold(store, seed, fold)
    if transform is not None:
        fold_data = transform(fold_data)
    out = []
    for name in components:
        X, Q = fold_data["views"][name]
        p = np.asarray(fit(name, X, fold_data["y_train"], Q), dtype=float)
        assert p.shape == (len(Q),) and np.isfinite(p).all()
        out.append(p)
    return seed, fold, np.column_stack(out)


def evaluate(store, fit=pipeline_fit, name=None, components=None, transform=None, jobs=1):
    """Fit `fit(view_name, X, y, Q) -> probabilities` on all fifteen cached folds.

    `components` picks views (default: all five). `transform(fold_data) -> fold_data`
    can add or replace views before fitting, e.g. to append a new feature block.
    Both must be top-level functions when `jobs > 1`.
    """
    store = Path(store)
    components = list(components or COMPONENTS)
    inputs = read_pickle(store / "inputs.pkl.gz")
    y, routing = inputs["y"], inputs["routing"]
    n_test = len(inputs["test_ids"])
    cv = np.full((len(SEEDS), len(y), len(components)), np.nan)
    test = np.full((len(SEEDS), n_test, len(components)), np.nan)
    tasks = [(seed, fold) for seed in SEEDS for fold in range(N_FOLDS)]
    args = (fit, components, transform)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_fit_fold, *zip(*[(store, s, f, *args) for s, f in tasks])))
    else:
        results = [_fit_fold(store, s, f, *args) for s, f in tasks]
    for seed, fold, p in results:
        repeat = SEEDS.index(seed)
        va = inputs["partitions"][seed][fold][1]
        cv[repeat, va] = p[: len(va)]
        test[repeat, np.flatnonzero(routing[seed] == fold)] = p[len(va) :]
    assert np.isfinite(cv).all() and np.isfinite(test).all()
    result = dict(names=np.asarray(components), y=y, cv=cv, test=test, test_ids=inputs["test_ids"])
    if name:
        (store / "experiments").mkdir(exist_ok=True)
        np.savez_compressed(store / "experiments" / f"{name}.npz", **result)
    return result


def load_result(store, name):
    with np.load(Path(store) / "experiments" / f"{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def mix(result, weights=None):
    """Blend components within each partition, then average the three partitions.

    `weights` maps component names to weights; the default is equal weights.
    Returns (repeat probabilities with shape (3, n), averaged probability).
    """
    names = list(result["names"])
    weights = weights or {n: 1 / len(names) for n in names}
    w = np.array([weights.get(n, 0.0) for n in names])
    repeated = result["cv"] @ w
    return repeated, repeated.mean(axis=0)


def merge(*results):
    """Put components from several saved runs side by side, for blending."""
    names = [n for r in results for n in r["names"]]
    assert len(set(names)) == len(names), "rename duplicate components first"
    return dict(
        names=np.asarray(names),
        y=results[0]["y"],
        cv=np.concatenate([r["cv"] for r in results], axis=2),
        test=np.concatenate([r["test"] for r in results], axis=2),
        test_ids=results[0]["test_ids"],
    )


def metrics(result, weights=None):
    y = result["y"]
    repeated, p = mix(result, weights)
    repeats = [float(accuracy_score(y, q >= THRESHOLD)) for q in repeated]
    return dict(
        repeat_accuracies=repeats,
        mean_repeat_accuracy=float(np.mean(repeats)),
        averaged_oof_accuracy=float(accuracy_score(y, p >= THRESHOLD)),
        averaged_oof_log_loss=float(log_loss(y, p)),
    )


def compare(new, old, store="outputs/store", new_weights=None, old_weights=None, draws=2000):
    """Paired comparison of averaged OOF predictions on the same battles.

    `new` and `old` are results or saved run names. The bootstrap interval
    resamples battles; it conditions on these fitted predictions and does not
    remove the selection bias of comparing many ideas on the same data.
    """
    new = load_result(store, new) if isinstance(new, str) else new
    old = load_result(store, old) if isinstance(old, str) else old
    y = new["y"]
    assert np.array_equal(y, old["y"])
    p_new, p_old = mix(new, new_weights)[1], mix(old, old_weights)[1]
    right_new, right_old = (p_new >= THRESHOLD) == y, (p_old >= THRESHOLD) == y
    diff = right_new.astype(float) - right_old
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(y), len(y))].mean() for _ in range(draws)]
    out = dict(
        accuracy_new=float(right_new.mean()),
        accuracy_old=float(right_old.mean()),
        accuracy_gain=float(diff.mean()),
        interval95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        fixes=int((right_new & ~right_old).sum()),
        breaks=int((~right_new & right_old).sum()),
        log_loss_new=float(log_loss(y, p_new)),
        log_loss_old=float(log_loss(y, p_old)),
        partition_gains=[
            float(accuracy_score(y, a >= THRESHOLD) - accuracy_score(y, b >= THRESHOLD))
            for a, b in zip(mix(new, new_weights)[0], mix(old, old_weights)[0])
        ],
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["reproduce"])
    parser.add_argument("--store", type=Path, default=Path("outputs/store"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    result = evaluate(args.store, name="reference", jobs=args.jobs)
    summary = metrics(result)
    summary["oof_components_sha256"] = array_digest(result["cv"])
    reference = json.loads((ROOT / "results" / "reference.json").read_text())
    for key in ["averaged_oof_accuracy", "mean_repeat_accuracy", "oof_components_sha256"]:
        summary[key + "_matches_reference"] = summary[key] == reference[key]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
