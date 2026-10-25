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

    # Control for E01: same starter features, but with scaling. E01 changes
    # both the features and the model, so we need this run to know how much
    # of the E01 gain is just the scaler.
    "e00b_starter_scaled": {
        "features": ["starter"],
        "model": "logreg_scaled",
    },

    # E01: add the state of the battle at turn 30 (faints, remaining HP,
    # statuses, active Pokemon). This is where we expect most of the signal.
    "e01_state_t30": {
        "features": ["starter", "state_t30"],
        "model": "logreg_scaled",
    },

    # E02: add the trajectory of the battle (HP difference along the way,
    # slope, momentum in the last turns, timing of the knockouts). Same
    # model as E01 so the comparison is only about the new features.
    "e02_trajectory": {
        "features": ["starter", "state_t30", "trajectory"],
        "model": "logreg_scaled",
    },
}
