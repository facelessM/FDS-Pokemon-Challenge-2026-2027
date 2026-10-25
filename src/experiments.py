"""
Experiment configurations.

One entry per experiment. An experiment is just a list of feature groups
plus a model name. We never edit an old entry after it has been logged,
we add a new one instead, so every number in experiments/log.md can be
reproduced with: python run.py --exp <name>
"""

EXPERIMENTS = {
    # Baseline: the starter notebook as it is (6 features, plain logistic
    # regression). Everything else gets compared against this.
    "e00_starter": {
        "features": ["starter"],
        "model": "logreg_starter",
    },
}
