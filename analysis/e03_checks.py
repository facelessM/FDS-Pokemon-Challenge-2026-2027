"""
Side checks for E03 (team strength and type matchups).

E03 did not improve on E01. Two questions:

  1. Is there any signal in the matchup features at all? We check them
     on their own, without the turn-30 state.
  2. If there is, is it already contained in the turn-30 state? A type
     advantage should show up as damage dealt, so by turn 30 its effect
     may already be visible in the HP numbers.

We also repeat the one-feature-at-a-time and gradient boosting checks
from E02.

    python analysis/e03_checks.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_species_table, find_data_dir, load_battles, load_type_chart
from src.features import build_features, get_target
from src.validation import N_SPLITS, SEED, cross_validate


def cv(X, y):
    r = cross_validate("logreg_scaled", X, y)
    return r["mean"], r["std"]


def main():
    data_dir = find_data_dir()
    train, _ = load_battles(data_dir)
    y = get_target(train)
    context = {"type_chart": load_type_chart(data_dir), "species": build_species_table(train)}

    X = build_features(train, ["starter", "state_t30", "matchup"], context).drop(columns="battle_id")
    e01_cols = list(build_features(train[:1], ["starter", "state_t30"]).columns[1:])
    match_cols = [c for c in X.columns if c not in e01_cols]

    # 1. Signal on their own, compared with the starter features (0.5164
    # with the same scaled model).
    print("Matchup features alone:      %.4f +/- %.4f" % cv(X[match_cols], y))
    print("Matchup + starter, no state: %.4f +/- %.4f" % cv(X[match_cols + e01_cols[:6]], y))

    # 2. How much of the turn-30 HP difference do the matchup features
    # explain? If a type edge turns into damage, it should correlate.
    print("\nCorrelation with diff_hp_total (turn 30):")
    print(X[match_cols].corrwith(X["diff_hp_total"]).sort_values().round(2).to_string())

    # One feature at a time on top of E01, in accuracy points.
    base, _ = cv(X[e01_cols], y)
    gains = {c: (cv(X[e01_cols + [c]], y)[0] - base) * 100 for c in match_cols}
    print(f"\nE01 baseline {base:.4f}; gain from adding one feature, in points:")
    print(pd.Series(gains).sort_values(ascending=False).round(2).to_string())

    # Non-linear model, default settings, same folds.
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for name, cols in [("E01 features", e01_cols), ("E03 features", e01_cols + match_cols)]:
        scores = cross_val_score(HistGradientBoostingClassifier(random_state=SEED),
                                 X[cols], y, cv=folds, scoring="accuracy")
        print(f"Gradient boosting, {name}: {scores.mean():.4f} +/- {scores.std():.4f}")


if __name__ == "__main__":
    main()
