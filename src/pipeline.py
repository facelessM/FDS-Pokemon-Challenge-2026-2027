"""Build inputs, fit the fifteen fold ensembles and collect their predictions."""

import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from . import features, raw, state
from .data import load_type_chart
from .forecast import CATS
from .models import lgbm_final, xgb_final
from .observations import describe, load, observation_features
from .species import snapshots
from .validation import COMPONENTS, N_FOLDS, SEEDS, THRESHOLD, blend, folds, route

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def signature(data, mode):
    """A cache belongs to exact inputs, source files and installed versions."""
    names = ["train.jsonl"]
    if mode == "predict":
        names += ["test.jsonl"]
    files = [data / name for name in names] + sorted(data.glob("*type_chart*.csv"))
    if mode == "predict":
        files += sorted(data.glob("sample_submission*.csv"))
    code = [
        ROOT / "run.py",
        ROOT / "requirements.txt",
        *sorted((ROOT / "src").glob("*.py")),
    ]
    detail = {
        "mode": mode,
        "data_sha256": {p.name: digest(p) for p in files},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code},
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }
    return (
        hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest(),
        detail,
    )


def save_pickle(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wb", compresslevel=2) as stream:
        pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def read_pickle(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def save_npz(path, **arrays):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def prefix_inputs(battles):
    rows, targets, levels = [], [], []
    for battle in battles:
        x, y, current = snapshots(battle)
        rows.extend(x)
        targets.append(y)
        levels.append(current)
    frame = pd.DataFrame(rows)
    for column in frame:
        if column not in CATS:
            frame[column] = frame[column].astype(np.float32)
    return dict(X=frame, target=np.asarray(targets), levels=np.asarray(levels))


def prefix_subset(prefixes, indices):
    rows = (5 * indices[:, None] + np.arange(5)).ravel()
    return dict(
        X=prefixes["X"].iloc[rows].reset_index(drop=True),
        target=prefixes["target"][indices],
        levels=prefixes["levels"][indices],
    )


def prepare(data, output, mode):
    key, provenance = signature(data, mode)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "inputs.pkl.gz"
    if path.exists():
        if read_pickle(path)["signature"] != key:
            raise ValueError("Inputs or source changed. Use a new --output directory.")
        return key
    train = load(data / "train.jsonl")
    test = load(data / "test.jsonl") if mode == "predict" else []
    assert train and all("player_won" in b for b in train)
    assert all("player_won" not in b for b in test)
    assert all(len(b["battle_timeline"]) == 30 for b in train + test)
    y = np.asarray([int(b["player_won"]) for b in train])
    assert set(y) == {0, 1}
    groups = np.asarray([describe(b)["group"] for b in train])
    test_groups = np.asarray([describe(b)["group"] for b in test])
    assignments, partitions, routing = {}, {}, {}
    for seed in SEEDS:
        partitions[seed], assignments[seed] = folds(y, groups, seed)
        routing[seed] = route(test_groups, seed, groups, assignments[seed])
        assert np.array_equal(
            routing[seed][::-1],
            route(test_groups[::-1], seed, groups, assignments[seed]),
        )
    ids = np.asarray([b["battle_id"] for b in test])
    train_ids = np.asarray([b["battle_id"] for b in train])
    assert len(np.unique(train_ids)) == len(train_ids)
    if test:
        sample = pd.read_csv(sorted(data.glob("sample_submission*.csv"))[0])
        assert sample.battle_id.is_unique and len(np.unique(ids)) == len(ids)
        assert set(ids) == set(sample.battle_id)
    inputs = dict(
        signature=key,
        provenance=provenance,
        y=y,
        groups=groups,
        test_groups=test_groups,
        train_ids=train_ids,
        test_ids=ids,
        assignments=assignments,
        partitions=partitions,
        routing=routing,
        raw=raw.frame([raw.summary(b) for b in train + test]),
        prefixes=prefix_inputs(train + test),
    )
    save_pickle(path, inputs)
    print(f"Prepared {len(train)} training and {len(test)} test battles.", flush=True)
    return key


def fit_fold(data, output, mode, seed, fold):
    started = time.time()
    key, _ = signature(data, mode)
    inputs = read_pickle(output / "inputs.pkl.gz")
    assert inputs["signature"] == key
    target = output / f"{seed}_{fold}.npz"
    if target.exists():
        with np.load(target) as saved:
            assert saved["signature"].item() == key
        return
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
    assert (labels[tr] >= 0).all() and (labels[n:] == -1).all()
    assert set(tr) | set(query) == set(range(len(battles)))
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
    print(f"{seed}/{fold}: reconstructed state", flush=True)
    original_indices = np.concatenate([np.arange(n), n + test_indices])
    prefixes = prefix_subset(inputs["prefixes"], original_indices)
    additional = features.additional(context, prefixes, tr, query, A, Z)
    matrices = features.components(A, Z, tr, query, additional)
    raw_frame = inputs["raw"].iloc[original_indices].reset_index(drop=True)
    del context, prefixes, inputs, additional, battles, views, train, test
    probabilities, columns = [], {}
    for name in COMPONENTS:
        if name == "raw_summary":
            X, Q = raw_frame.iloc[tr].copy(), raw_frame.iloc[query].copy()
            for column in X:
                if column.startswith("c_"):
                    levels = sorted(X[column].unique())
                    X[column] = pd.Categorical(X[column], categories=levels)
                    Q[column] = pd.Categorical(Q[column], categories=levels)
            model = lgbm_final().fit(X, labels[tr])
            p = model.predict_proba(Q)[:, 1]
            del model
        else:
            X, Q = matrices[name]
            estimates = []
            for factory in [lgbm_final, xgb_final]:
                model = factory().fit(X, labels[tr])
                estimates.append(model.predict_proba(Q)[:, 1])
                del model
            p = (estimates[0] + estimates[1]) / 2
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        columns[name] = X.shape[1]
        probabilities.append(p)
        print(f"{seed}/{fold}: {name}, {X.shape[1]} columns", flush=True)
    probabilities = np.column_stack(probabilities)
    save_npz(
        target,
        signature=key,
        tr=tr,
        va=va,
        y=labels[va],
        test_indices=test_indices,
        names=np.asarray(COMPONENTS),
        cv_components=probabilities[: len(va)],
        test_components=probabilities[len(va) :],
    )
    (output / f"{seed}_{fold}.json").write_text(
        json.dumps(
            dict(
                signature=key,
                seed=seed,
                fold=fold,
                training_rows=len(tr),
                validation_rows=len(va),
                test_rows=len(test_indices),
                columns=columns,
                seconds=round(time.time() - started, 1),
            ),
            indent=2,
        )
    )
    print(f"{seed}/{fold}: saved in {time.time()-started:.1f}s", flush=True)


def run_folds(data, output, mode, jobs):
    """Keep parallelism bounded; each child releases its feature matrices on exit."""
    pending = [(seed, fold) for seed in SEEDS for fold in range(N_FOLDS)]
    active = []
    try:
        while pending or active:
            while pending and len(active) < jobs:
                seed, fold = pending.pop(0)
                log_path = output / f"{seed}_{fold}.log"
                stream = log_path.open("a")
                command = [
                    sys.executable,
                    str(ROOT / "run.py"),
                    mode,
                    "--data",
                    str(data),
                    "--output",
                    str(output),
                    "--worker",
                    str(seed),
                    str(fold),
                ]
                child = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                active.append((child, stream, seed, fold, log_path))
                print(f"Started {seed}/{fold}", flush=True)
            for entry in active[:]:
                child, stream, seed, fold, log_path = entry
                status = child.poll()
                if status is None:
                    continue
                stream.close()
                active.remove(entry)
                if status:
                    raise RuntimeError(f"Fold {seed}/{fold} failed; see {log_path}")
                print(f"Finished {seed}/{fold}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for child, stream, *_ in active:
            child.terminate()
            child.wait()
            stream.close()


def collect(data, output, mode):
    key, provenance = signature(data, mode)
    inputs = read_pickle(output / "inputs.pkl.gz")
    assert inputs["signature"] == key
    y, ids = inputs["y"], inputs["test_ids"]
    cv = np.full((len(SEEDS), len(y), len(COMPONENTS)), np.nan)
    test = np.full((len(SEEDS), len(ids), len(COMPONENTS)), np.nan)
    counts_cv = np.zeros(cv.shape[:2], int)
    counts_test = np.zeros(test.shape[:2], int)
    for repeat, seed in enumerate(SEEDS):
        for fold in range(N_FOLDS):
            with np.load(output / f"{seed}_{fold}.npz") as result:
                assert result["signature"].item() == key
                assert result["names"].tolist() == COMPONENTS
                tr, va = inputs["partitions"][seed][fold]
                target = np.flatnonzero(inputs["routing"][seed] == fold)
                assert np.array_equal(result["tr"], tr) and np.array_equal(
                    result["va"], va
                )
                assert np.array_equal(result["y"], y[va])
                assert np.array_equal(result["test_indices"], target)
                cv[repeat, va] = result["cv_components"]
                test[repeat, target] = result["test_components"]
                counts_cv[repeat, va] += 1
                counts_test[repeat, target] += 1
    assert (counts_cv == 1).all() and (counts_test == 1).all()
    assert np.isfinite(cv).all() and np.isfinite(test).all()
    repeated = blend(cv)
    probability = repeated.mean(axis=0)
    baseline = cv[:, :, 0].mean(axis=0)
    repeats = [float(accuracy_score(y, p >= THRESHOLD)) for p in repeated]
    results = dict(
        seeds=SEEDS,
        component_order=COMPONENTS,
        threshold=THRESHOLD,
        repeat_accuracies=repeats,
        mean_repeat_accuracy=float(np.mean(repeats)),
        averaged_oof_accuracy=float(accuracy_score(y, probability >= THRESHOLD)),
        averaged_oof_log_loss=float(log_loss(y, probability)),
        baseline_averaged_oof_accuracy=float(accuracy_score(y, baseline >= THRESHOLD)),
        oof_components_sha256=array_digest(cv),
        oof_probability_sha256=array_digest(probability),
        provenance=provenance,
        interpretation="Repeated validation of a method selected on these battles, not an independent holdout.",
    )
    save_npz(
        output / "oof.npz",
        y=y,
        battle_id=inputs["train_ids"],
        groups=inputs["groups"],
        components=cv,
        repeat_probabilities=repeated,
        p=probability,
    )
    if mode == "predict":
        test_repeats = blend(test)
        test_probability = test_repeats.mean(axis=0)
        sample = pd.read_csv(sorted(data.glob("sample_submission*.csv"))[0])
        ordered = (
            pd.Series(test_probability, index=ids).loc[sample.battle_id].to_numpy()
        )
        submission = pd.DataFrame(
            dict(
                battle_id=sample.battle_id,
                player_won=(ordered >= THRESHOLD).astype(int),
            )
        )
        assert len(submission) == len(ids) and submission.battle_id.is_unique
        assert not submission.isna().any().any()
        path = output / "submission_matched_core_summary.csv"
        submission.to_csv(path, index=False)
        save_npz(
            output / "test_probabilities.npz",
            battle_id=ids,
            components=test,
            repeat_probabilities=test_repeats,
            p=test_probability,
        )
        results.update(
            submission_rows=len(submission),
            submission_sha256=digest(path),
            test_components_sha256=array_digest(test),
            test_probability_sha256=array_digest(test_probability),
            matching_train_test_timeline_rows=int(
                np.isin(inputs["test_groups"], inputs["groups"]).sum()
            ),
        )
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in results.items() if k != "provenance"}, indent=2),
        flush=True,
    )
    return results
