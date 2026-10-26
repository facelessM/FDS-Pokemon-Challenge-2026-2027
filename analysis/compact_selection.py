"""
How the columns of the compact model were chosen, and what it scores.

  1. Build the final feature groups plus `roles` (Spikes, Rapid Spin and
     Explosion users).
  2. Fit LightGBM on 80% of the training battles and, on the other 20%,
     shuffle one column at a time (permutation importance, 5 repeats). A
     column is kept if shuffling it makes the predictions worse on average.
     The list is written to experiments/compact_columns.txt.
  3. Compare the compact model with the final model on CV seeds 6 to 10.

    python analysis/compact_selection.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target, roles_features
from src.models import lgbm_final
from src.validation import cross_validate

SEEDS = [6, 7, 8, 9, 10]


def main():
    data_dir = find_data_dir()
    train, _ = load_battles(data_dir)
    y = get_target(train)
    context = {
        "type_chart": load_type_chart(data_dir),
        "species": build_species_table(train),
        "move_vocab": build_move_vocab(train),
        "top_species": build_top_species(train),
    }
    X = build_features(train, EXPERIMENTS["compact_blend"]["features"], context).drop(columns="battle_id")

    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = lgbm_final().fit(X_fit, y_fit)
    imp = permutation_importance(model, X_val, y_val, scoring="neg_log_loss", n_repeats=5, random_state=0)
    keep = [c for c, v in zip(X.columns, imp.importances_mean) if v > 0]
    (ROOT / EXPERIMENTS["compact_blend"]["columns"]).write_text("\n".join(keep) + "\n")

    role_cols = list(roles_features(train[0]))
    print(f"Kept {len(keep)} of {X.shape[1]} columns")
    for name, test in [("move usage", lambda c: "_mv_" in c),
                       ("species still standing", lambda c: "_alive_" in c),
                       ("roles", lambda c: c in role_cols)]:
        print(f"  {name:24s} {sum(test(c) for c in keep)}")

    X_final = X.drop(columns=role_cols)
    for label, M in [("final model", X_final), ("compact model", X[keep])]:
        scores = [cross_validate("lgb_xgb_blend", M, y, seed=s)["mean"] for s in SEEDS]
        print(f"{label:14s} {M.shape[1]:3d} columns, seeds 6-10: {np.mean(scores):.4f} +/- {np.std(scores):.4f}  "
              f"({', '.join(f'{s:.4f}' for s in scores)})")


if __name__ == "__main__":
    main()
