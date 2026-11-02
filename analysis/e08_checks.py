"""
Diagnostics for E08 (uncertainty about Player 2's hidden information).

We check:
  1. Feature distributions: how many battles have p2_unseen_count > 0, > 2, > 4.
  2. Correlation between uncertainty features and model residuals (outcome - predicted_prob).
  3. Accuracy on close battles (baseline prob within 0.10 of 0.5) with and without uncertainty features.
  4. Permutation importance of the new uncertainty features on validation data.

    python analysis/e08_checks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (build_matchup_chart, build_move_vocab, build_species_table,
                      build_team_index, build_top_species, find_data_dir,
                      load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target, uncertainty_features
from src.models import get_model
from src.validation import SEED, cross_validate


def main():
    data_dir = find_data_dir()
    train, _ = load_battles(data_dir)
    y = get_target(train)
    context = {
        "type_chart": load_type_chart(data_dir),
        "species": build_species_table(train),
        "move_vocab": build_move_vocab(train),
        "top_species": build_top_species(train),
        "teams": build_team_index(train),
        "matchups": build_matchup_chart(train),
    }

    # 1. Feature distributions
    print("1. Uncertainty Feature Distributions:")
    u_rows = [uncertainty_features(b) for b in train]
    df_u = pd.DataFrame(u_rows)
    u_cols = list(df_u.columns)

    n = len(train)
    p2_unseen = df_u["p2_unseen_count"]
    print(f"   Battles with p2_unseen_count > 0: {(p2_unseen > 0).sum()} / {n} ({(p2_unseen > 0).mean():.1%})")
    print(f"   Battles with p2_unseen_count > 2: {(p2_unseen > 2).sum()} / {n} ({(p2_unseen > 2).mean():.1%})")
    print(f"   Battles with p2_unseen_count > 4: {(p2_unseen > 4).sum()} / {n} ({(p2_unseen > 4).mean():.1%})")
    print(f"   All 6 P2 Pokemon known:           {(p2_unseen == 0).sum()} / {n} ({(p2_unseen == 0).mean():.1%})")
    print("\n   Summary statistics:")
    print(df_u.describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

    # 2. Baseline Model & Residual Correlations
    print("\n2. Baseline (Team) Out-of-Fold Predictions & Residual Correlations:")
    base_config = EXPERIMENTS["team_blend"]
    X_base = build_features(train, base_config["features"], context).drop(columns="battle_id")
    cv_base = cross_validate(base_config["model"], X_base, y, seed=SEED)
    p_base = cv_base["oof_proba"]
    residuals = y - p_base

    corrs = {}
    for col in u_cols:
        r = np.corrcoef(df_u[col], residuals)[0, 1]
        corrs[col] = r
    corr_series = pd.Series(corrs).sort_values(key=abs, ascending=False)
    print("   Correlation with residuals (outcome - baseline_prob):")
    for col, r in corr_series.items():
        print(f"     {col:32s}: r = {r:+.4f}")

    # 3. Accuracy on close battles
    print("\n3. Accuracy on Close Battles (baseline predicted prob within 0.10 of 0.5):")
    close_mask = np.abs(p_base - 0.5) <= 0.10
    n_close = np.sum(close_mask)
    base_close_acc = np.mean((p_base[close_mask] >= 0.5) == y[close_mask])
    base_overall_acc = cv_base["mean"]

    e08_config = EXPERIMENTS["e08_uncertainty"]
    X_e08 = build_features(train, e08_config["features"], context).drop(columns="battle_id")
    cv_e08 = cross_validate(e08_config["model"], X_e08, y, seed=SEED)
    p_e08 = cv_e08["oof_proba"]
    e08_close_acc = np.mean((p_e08[close_mask] >= 0.5) == y[close_mask])
    e08_overall_acc = cv_e08["mean"]

    print(f"   Total close battles: {n_close} / {n} ({n_close / n:.1%})")
    print(f"   Baseline overall CV accuracy:       {base_overall_acc:.4f} +/- {cv_base['std']:.4f}")
    print(f"   e08_uncertainty overall CV accuracy: {e08_overall_acc:.4f} +/- {cv_e08['std']:.4f} (diff: {e08_overall_acc - base_overall_acc:+.4f})")
    print(f"   Baseline close battles accuracy:    {base_close_acc:.4f}")
    print(f"   e08 close battles accuracy:         {e08_close_acc:.4f} (diff: {e08_close_acc - base_close_acc:+.4f})")

    # 4. Permutation Importance
    print("\n4. Permutation Importance of Uncertainty Features (Validation fold 0):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(X_e08, y))

    model = get_model(e08_config["model"], SEED).fit(X_e08.iloc[train_idx], y[train_idx])
    val_X = X_e08.iloc[val_idx].copy()
    val_y = y[val_idx]
    base_val_acc = accuracy_score(val_y, (model.predict_proba(val_X)[:, 1] >= 0.5).astype(int))
    print(f"   Fold 0 validation accuracy: {base_val_acc:.4f}")

    rng = np.random.RandomState(SEED)
    importances = {}
    for col in u_cols:
        orig = val_X[col].copy()
        drops = []
        for _ in range(5):
            val_X[col] = rng.permutation(orig)
            p_shuff = model.predict_proba(val_X)[:, 1]
            acc_shuff = accuracy_score(val_y, (p_shuff >= 0.5).astype(int))
            drops.append(base_val_acc - acc_shuff)
        val_X[col] = orig
        importances[col] = np.mean(drops)

    imp_series = pd.Series(importances).sort_values(ascending=False)
    for col, imp in imp_series.items():
        print(f"     {col:32s}: importance = {imp:+.5f}")


if __name__ == "__main__":
    main()
