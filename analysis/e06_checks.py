"""
Side checks for E06 (model comparison).

The three main models end up within noise of each other. Before we decide
how to use them, we check whether they make the same mistakes. If they do,
combining them later (E09) cannot help much; if they disagree on a fair
share of battles, it might.

    python analysis/e06_checks.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import find_data_dir, load_battles
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.validation import cross_validate

MODELS = ["logreg_scaled", "random_forest", "gb_default", "knn_scaled"]


def main():
    train, _ = load_battles(find_data_dir())
    y = get_target(train)
    X = build_features(train, EXPERIMENTS["e05_sacrifice"]["features"]).drop(columns="battle_id")

    # Out-of-fold win probabilities for each model, same folds as always.
    proba = {m: cross_validate(m, X, y)["oof_proba"] for m in MODELS}
    pred = {m: (p >= 0.5).astype(int) for m, p in proba.items()}

    print("Pairwise comparison of out-of-fold predictions")
    rows = []
    for a, b in combinations(MODELS, 2):
        rows.append({
            "models": f"{a} / {b}",
            "proba_corr": np.corrcoef(proba[a], proba[b])[0, 1],
            "disagree": (pred[a] != pred[b]).mean(),
            # Of the battles where they disagree, how often is the first one right?
            "first_right_when_disagree": (pred[a][pred[a] != pred[b]] == y[pred[a] != pred[b]]).mean(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Share of battles that every model gets wrong: an upper limit on what
    # any combination of these models could fix.
    all_wrong = np.all([pred[m] != y for m in MODELS[:3]], axis=0).mean()
    print(f"\nBattles all three main models get wrong: {all_wrong:.1%}")


if __name__ == "__main__":
    main()
