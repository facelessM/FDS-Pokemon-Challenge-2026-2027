"""
Side checks for E01 (battle state at turn 30).

These are not experiments of their own. They answer the questions we had
while reading the E01 result, using the same folds, and every number
quoted in the E01 entry of experiments/log.md comes from here.

    python analysis/e01_checks.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import find_data_dir, load_battles
from src.features import build_features, get_target
from src.models import get_model
from src.validation import SEED, cross_validate


def cv_score(X, y):
    r = cross_validate("logreg_scaled", X, y)
    return f"{r['mean']:.4f} +/- {r['std']:.4f}"


def main():
    train, _ = load_battles(find_data_dir())
    y = get_target(train)
    X = build_features(train, ["starter", "state_t30"]).drop(columns="battle_id")

    all_cols = list(X.columns)
    starter_cols = [c for c in all_cols if c.startswith(("p1_mean", "p2_lead"))]
    state_cols = [c for c in all_cols if c not in starter_cols]

    print("Full E01 set:                ", cv_score(X, y))
    print("Without the starter features:", cv_score(X[state_cols], y))
    print("Team HP difference only:     ", cv_score(X[["diff_hp_total"]], y))
    print("Faint counts only:           ", cv_score(X[["p1_n_fainted", "p2_n_fainted"]], y))

    # What if we only count the HP of P2 Pokemon we have actually seen,
    # instead of assuming unseen ones are at full HP?
    alt = X.copy()
    p2_unseen = (6 - alt["p2_n_seen"]).clip(lower=0)
    alt["p2_hp_total"] = alt["p2_hp_total"] - p2_unseen
    alt["diff_hp_total"] = alt["p1_hp_total"] - alt["p2_hp_total"]
    print("Unseen P2 not counted:       ", cv_score(alt, y))

    # Coefficients of the scaled model trained on everything. Since the
    # inputs are standardised, their size is comparable across features.
    model = get_model("logreg_scaled", SEED).fit(X, y)
    coefs = pd.Series(model[-1].coef_[0], index=all_cols)
    print("\nLargest standardised coefficients:")
    print(coefs.sort_values(key=abs, ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
