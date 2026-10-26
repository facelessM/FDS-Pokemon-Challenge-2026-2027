"""
Model definitions.

Only classical ML is allowed in this competition (no neural networks),
so everything here comes from scikit-learn or a gradient boosting
library. Each entry in MODELS is a function that takes a random seed
and returns a fresh, untrained estimator. We build a new one for every
fold so nothing leaks between folds.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logreg_starter(seed):
    """Exactly the model from the starter notebook (no scaling)."""
    return LogisticRegression(random_state=seed, max_iter=1000)


def logreg_scaled(seed):
    """Logistic regression with standardised inputs.

    Features end up on very different scales (base stats in the 100s,
    HP fractions between 0 and 1), and an unscaled logistic regression
    handles that badly. Scaling first is the usual fix.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed, max_iter=2000),
    )


def gb_default(seed):
    """Gradient boosting with scikit-learn's default settings.

    Not tuned on purpose. We run it next to every experiment as a check:
    if a feature group only helps a non-linear model, this is where it
    shows up.
    """
    return HistGradientBoostingClassifier(random_state=seed)


def knn_scaled(seed):
    """k-nearest neighbours: a battle is predicted like the most similar ones.

    Distances only make sense on scaled inputs. k = 89 is the usual
    square-root rule for the 8,000 battles in a training fold. The seed is
    not used (k-NN has no randomness) but we keep the same signature.
    """
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=89))


def random_forest(seed):
    """Random forest with default settings, apart from the number of trees.

    300 trees instead of the default 100, so the result does not depend
    much on the random draw. Trees do not need scaled inputs.
    """
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def lgbm_final(seed=0):
    """LightGBM half of the final blend: small trees (15 leaves), a slow
    learning rate, row and column subsampling. Seed fixed at 0."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
                              n_jobs=2, verbose=-1)


def xgb_final(seed=0):
    """XGBoost half of the final blend, with matching mild settings
    (depth 4). Seed fixed at 0."""
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                             reg_lambda=1.0, random_state=0, n_jobs=2, verbosity=0)


class LgbXgbBlend:
    """LightGBM and XGBoost trained side by side, win probabilities averaged.

    Both seeds are fixed at 0, the values behind our submitted file.
    """

    def __init__(self):
        self.models = [lgbm_final(), xgb_final()]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def lgb_xgb_blend(seed):
    """The final model. The seed argument is unused (see LgbXgbBlend)."""
    return LgbXgbBlend()


class TeamRecordModel:
    """Adds each team's past record to the features, then fits a model.

    Teams recur across battles, and a team that won often before tends to
    win again. The record of a team is its smoothed win rate in the
    training rows: a battle counts for P1's team with its outcome, and for
    P2's team (when the Pokemon seen match a single known team) with the
    opposite outcome. Records are computed inside fit, from the rows the
    model is trained on, so cross validation stays honest. The training
    rows get theirs from an inner 5-fold split, so a battle never sees its
    own result.
    """

    ID_COLS = ["p1_team_id", "p2_team_id"]
    SMOOTHING = 10   # a team needs about 10 battles before its record counts as much as the 50% prior

    def __init__(self, make_model):
        self.make_model = make_model

    def _stats(self, ids, y):
        wins, games = defaultdict(float), defaultdict(float)
        for p1, p2, cand, won in zip(ids["p1_team_id"], ids["p2_team_id"], ids["p2_team_candidates"], y):
            wins[p1] += won
            games[p1] += 1
            if p2 >= 0 and cand == 1:
                wins[p2] += 1 - won
                games[p2] += 1
        return wins, games

    def _records(self, ids, stats):
        wins, games = stats
        m = self.SMOOTHING
        rows = []
        for p1, p2 in zip(ids["p1_team_id"], ids["p2_team_id"]):
            g1 = games.get(p1, 0) if p1 >= 0 else 0
            g2 = games.get(p2, 0) if p2 >= 0 else 0
            r1 = (wins.get(p1, 0) + 0.5 * m) / (g1 + m)
            r2 = (wins.get(p2, 0) + 0.5 * m) / (g2 + m)
            rows.append((r1, g1, r2, g2, r1 - r2))
        return pd.DataFrame(rows, columns=["p1_team_record", "p1_team_battles", "p2_team_record",
                                           "p2_team_battles", "diff_team_record"], index=ids.index)

    def _with_records(self, X, records):
        return pd.concat([X.drop(columns=self.ID_COLS), records], axis=1)

    def fit(self, X, y):
        y = np.asarray(y)
        ids = X[self.ID_COLS + ["p2_team_candidates"]]
        inner = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=99).split(X, y):
            inner.append(self._records(ids.iloc[b], self._stats(ids.iloc[a], y[a])))
        records = pd.concat(inner).loc[X.index]
        self.stats_ = self._stats(ids, y)
        self.model_ = self.make_model().fit(self._with_records(X, records), y)
        return self

    def predict_proba(self, X):
        ids = X[self.ID_COLS + ["p2_team_candidates"]]
        return self.model_.predict_proba(self._with_records(X, self._records(ids, self.stats_)))


def team_lgb_xgb_blend(seed):
    """The final blend with team records added."""
    return TeamRecordModel(LgbXgbBlend)


MODELS = {
    "logreg_starter": logreg_starter,
    "logreg_scaled": logreg_scaled,
    "gb_default": gb_default,
    "knn_scaled": knn_scaled,
    "random_forest": random_forest,
    "lgb_xgb_blend": lgb_xgb_blend,
    "lgbm_final": lgbm_final,
    "xgb_final": xgb_final,
    "team_lgb_xgb_blend": team_lgb_xgb_blend,
}


def get_model(name, seed):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](seed)
