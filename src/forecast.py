"""Cross-fit five-turn HP changes using only transitions inside turns 1 to 30."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CATS = ["p1_species", "p1_last_move", "p2_species", "p2_last_move"]


def forecaster():
    return lgb.LGBMRegressor(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=5,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        random_state=27,
        n_jobs=2,
        verbosity=-1,
    )


def fit_forecast(z, fit, target):
    # Four prefix examples per training battle; prediction uses turns 25 and 30.
    train_rows = (5 * fit[:, None] + np.arange(4)).ravel()
    query_rows = (5 * target[:, None] + np.array([3, 4])).ravel()
    X, Q = z["X"].iloc[train_rows].copy(), z["X"].iloc[query_rows].copy()
    for c in CATS:
        levels = sorted(X[c].unique())
        X[c] = pd.Categorical(X[c], categories=levels)
        Q[c] = pd.Categorical(Q[c], categories=levels)
    y = z["target"][fit].reshape(-1, 2)
    pred = np.stack([forecaster().fit(X, y[:, j]).predict(Q) for j in range(2)], axis=1)
    assert np.isfinite(pred).all()
    return pred.reshape(len(target), 2, 2)


def crossfit_forecast(z, y, groups, tr, va):
    q = np.zeros((len(y), 2, 2))
    assigned = np.zeros(len(y), int)
    for a, b in StratifiedGroupKFold(2, shuffle=True, random_state=27183).split(
        np.zeros(len(tr)), y[tr], groups[tr]
    ):
        fit, query = tr[a], tr[b]
        assert not set(groups[fit]) & set(groups[query])
        q[query] = fit_forecast(z, fit, query)
        assigned[query] += 1
    assert not set(groups[tr]) & set(groups[va])
    q[va] = fit_forecast(z, tr, va)
    assigned[va] += 1
    assert (assigned == 1).all()
    actual = z["target"][:, 3, :]
    current = z["levels"][:, 4, :]
    frames = {}
    for j, side in enumerate(["p1", "p2"]):
        frames[f"forecast_{side}_next_delta"] = q[:, 1, j]
        frames[f"forecast_{side}_next_hp"] = current[:, j] + q[:, 1, j]
        frames[f"forecast_{side}_last5_surprise"] = actual[:, j] - q[:, 0, j]
    for key in ["next_delta", "next_hp", "last5_surprise"]:
        frames["forecast_diff_" + key] = (
            frames["forecast_p1_" + key] - frames["forecast_p2_" + key]
        )
    mean = z["target"][tr].mean(axis=(0, 1))
    metrics = {
        "validation_turn25_to30_mse": ((actual[va] - q[va, 0]) ** 2)
        .mean(axis=0)
        .tolist(),
        "zero_change_mse": (actual[va] ** 2).mean(axis=0).tolist(),
        "training_mean_mse": ((actual[va] - mean) ** 2).mean(axis=0).tolist(),
        "side_order": ["p1", "p2"],
    }
    return pd.DataFrame(frames).astype(np.float32), metrics
