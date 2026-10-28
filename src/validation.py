"""Grouped folds and label-free routing for the submitted estimator."""

import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = [20261026, 20261105, 193]
N_FOLDS = 5
COMPONENTS = ["baseline", "species_forecast", "tactical", "zero_power", "raw_summary"]
WEIGHTS = [0.2] * 5
THRESHOLD = 0.5


def folds(y, groups, seed):
    splits = list(
        StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, groups
        )
    )
    assigned = np.full(len(y), -1, dtype=int)
    for fold, (train, valid) in enumerate(splits):
        assert not set(groups[train]) & set(groups[valid])
        assigned[valid] = fold
    assert (assigned >= 0).all()
    return splits, assigned


def route(groups, seed, train_groups, train_folds):
    """A known timeline uses its held-out fold; new groups use a fixed hash."""
    mapping = {}
    for group, fold in zip(train_groups, train_folds):
        group = str(group)
        assert group not in mapping or mapping[group] == int(fold)
        mapping[group] = int(fold)
    assigned = []
    for group in groups:
        group = str(group)
        if group in mapping:
            assigned.append(mapping[group])
        else:
            key = f"matched-core-summary-v1|{seed}|{group}".encode()
            assigned.append(int(hashlib.sha256(key).hexdigest(), 16) % N_FOLDS)
    return np.asarray(assigned, dtype=int)


def blend(probabilities):
    """Average probabilities first. Apply the threshold only in the final step."""
    assert probabilities.shape[-1] == len(WEIGHTS)
    return sum(weight * probabilities[..., j] for j, weight in enumerate(WEIGHTS))
