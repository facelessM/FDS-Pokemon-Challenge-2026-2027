"""Fixed settings used in every outer fold."""


def lgbm_final(seed=0):
    """LightGBM half of the final blend: small trees (15 leaves), a slow
    learning rate, row and column subsampling. Seed fixed at 0."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=2,
        verbose=-1,
    )


def xgb_final(seed=0):
    """XGBoost half of the final blend, with matching mild settings
    (depth 4). Seed fixed at 0."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=2,
        verbosity=0,
    )
