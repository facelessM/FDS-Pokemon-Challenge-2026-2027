"""
Side checks for E02 (battle trajectory).

E02 did not improve on E01, so we tested three possible reasons, each on
the same folds as the main experiments:

  H1  the trajectory only repeats what the turn-30 state already says
  H2  momentum matters, but only in combination with other features,
      which a linear model cannot pick up
  H3  a few trajectory features help but get lost among the others

    python analysis/e02_checks.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import find_data_dir, load_battles
from src.features import build_features, get_target
from src.validation import N_SPLITS, SEED, cross_validate


def cv_mean(X, y):
    return cross_validate("logreg_scaled", X, y)["mean"]


def main():
    train, _ = load_battles(find_data_dir())
    y = get_target(train)
    X = build_features(train, ["starter", "state_t30", "trajectory"]).drop(columns="battle_id")

    # Split the columns back into "what E01 had" and "what E02 added".
    e01_cols = list(build_features(train[:1], ["starter", "state_t30"]).columns[1:])
    traj_cols = [c for c in X.columns if c not in e01_cols]

    # H1: how much do the trajectory features know on their own, and how
    # closely do they follow the final HP difference?
    r = cross_validate("logreg_scaled", X[traj_cols], y)
    print(f"H1  trajectory features alone: {r['mean']:.4f} +/- {r['std']:.4f}")
    print("    correlation with diff_hp_total (turn 30):")
    corr = X[traj_cols].corrwith(X["diff_hp_total"]).sort_values()
    print(corr.round(2).to_string())

    # H2: does a non-linear model get anything out of the trajectory?
    # Default settings on purpose, this is a quick check and not tuning.
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for name, cols in [("E01 features", e01_cols), ("E02 features", e01_cols + traj_cols)]:
        model = HistGradientBoostingClassifier(random_state=SEED)
        scores = cross_val_score(model, X[cols], y, cv=folds, scoring="accuracy")
        print(f"H2  gradient boosting, {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # H3: add each trajectory feature to E01 on its own and see how much
    # the CV mean moves (in accuracy points).
    base = cv_mean(X[e01_cols], y)
    gains = {c: (cv_mean(X[e01_cols + [c]], y) - base) * 100 for c in traj_cols}
    print(f"H3  E01 baseline {base:.4f}; gain from adding one feature, in points:")
    print(pd.Series(gains).sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
