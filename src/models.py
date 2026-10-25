"""
Model definitions.

Only classical ML is allowed in this competition (no neural networks),
so everything here comes from scikit-learn or a gradient boosting
library. Each entry in MODELS is a function that takes a random seed
and returns a fresh, untrained estimator. We build a new one for every
fold so nothing leaks between folds.
"""

from sklearn.linear_model import LogisticRegression
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


MODELS = {
    "logreg_starter": logreg_starter,
    "logreg_scaled": logreg_scaled,
}


def get_model(name, seed):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](seed)
