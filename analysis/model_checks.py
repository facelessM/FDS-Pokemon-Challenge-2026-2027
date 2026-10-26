"""
Model choices behind the final submission, on the usual folds (seed 42).

  0. LightGBM on the E05 features, as a reference.
  1. The final features with each model on its own: logistic regression,
     untuned gradient boosting, LightGBM, XGBoost, and the blend of the
     last two.
  2. A random search over 20 LightGBM settings, to see whether tuning
     beats the mild settings we use.

    python analysis/model_checks.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform, randint, uniform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models
from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.validation import cross_validate


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
    X = build_features(train, EXPERIMENTS["final_blend"]["features"], context).drop(columns="battle_id")

    # Same LightGBM on the E05 features, so the gain from the new feature
    # groups can be told apart from the gain from changing the model.
    X05 = build_features(train, EXPERIMENTS["e05_sacrifice"]["features"], context).drop(columns="battle_id")
    r = cross_validate("lgbm_final", X05, y)
    print(f"0. LightGBM on the E05 features ({X05.shape[1]} columns): {r['mean']:.4f} +/- {r['std']:.4f}\n")

    print(f"1. Final features ({X.shape[1]} columns), one model at a time")
    for name in ["logreg_scaled", "gb_default", "lgbm_final", "xgb_final", "lgb_xgb_blend"]:
        r = cross_validate(name, X, y)
        print(f"   {name:14s} {r['mean']:.4f} +/- {r['std']:.4f}")

    # 2. Random search around the LightGBM settings. Each sampled setting is
    # registered as a temporary model so it runs through the same CV code.
    space = {
        "n_estimators": randint(200, 1300),
        "learning_rate": loguniform(0.01, 0.08),
        "num_leaves": randint(4, 41),
        "min_child_samples": randint(10, 201),
        "colsample_bytree": uniform(0.1, 0.7),
        "subsample": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-3, 30),
    }
    scores = []
    for params in ParameterSampler(space, n_iter=20, random_state=0):
        models.MODELS["_search"] = lambda seed, p=params: lgb.LGBMClassifier(
            subsample_freq=1, random_state=0, n_jobs=2, verbose=-1, **p)
        scores.append(cross_validate("_search", X, y)["mean"])
    print(f"\n2. Random search, 20 LightGBM settings: best {max(scores):.4f}, "
          f"median {np.median(scores):.4f}")


if __name__ == "__main__":
    main()
