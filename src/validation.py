"""
Local validation.

The leaderboard only shows part of the test set and we get 3 submissions
a day, so every decision (which features, which model, which settings)
is made on k-fold cross validation over train.jsonl.

We always use the same seed, so the folds are identical across
experiments. That matters: if the folds changed between two runs we
could not tell whether a gain came from our change or from a luckier
split.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .models import get_model


N_SPLITS = 5
SEED = 42


def cross_validate(model_name, X, y, n_splits=N_SPLITS, seed=SEED):
    """Run stratified k-fold CV and return the scores.

    Stratified means each fold keeps the same win/loss ratio as the full
    training set (it is 50/50 here, but it is a good habit anyway).

    Returns a dict with:
      fold_scores : accuracy on each held out fold
      mean, std   : summary of fold_scores
      oof_proba   : out-of-fold win probability for every training battle.
                    We keep these for error analysis and, later on, for
                    ensembling.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))
    fold_scores = []

    for train_idx, valid_idx in skf.split(X, y):
        model = get_model(model_name, seed)
        model.fit(X.iloc[train_idx], y[train_idx])

        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof_proba[valid_idx] = proba
        fold_scores.append(accuracy_score(y[valid_idx], (proba >= 0.5).astype(int)))

    return {
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "oof_proba": oof_proba,
    }
