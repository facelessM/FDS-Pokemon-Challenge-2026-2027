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

    # E03: add team strength and type matchups. Built on E01, not E02,
    # because the trajectory group was dropped.
    "e03_matchup": {
        "features": ["starter", "state_t30", "matchup"],
        "model": "logreg_scaled",
    },

    # E05: tell sacrifices (Explosion, Self-Destruct) apart from knockouts
    # taken. Came out of the error analysis in E04.
    "e05_sacrifice": {
        "features": ["starter", "state_t30", "sacrifice"],
        "model": "logreg_scaled",
    },

    # Final model: E05 plus what each side did (move usage and the realised
    # effect of the moves) and which common species are still standing, with
    # a LightGBM + XGBoost blend. Later models build on these feature groups.
    "final_blend": {
        "features": ["starter", "state_t30", "sacrifice",
                     "move_usage", "move_effects", "alive_species"],
        "model": "lgb_xgb_blend",
    },

    # Compact model: the final groups plus three Gen 2 roles (Spikes, Rapid
    # Spin, Explosion), then only the columns that the selection in
    # analysis/compact_selection.py found useful. Same blend as the final.
    "compact_blend": {
        "features": ["starter", "state_t30", "sacrifice",
                     "move_usage", "move_effects", "alive_species", "roles"],
        "columns": "experiments/compact_columns.txt",
        "model": "lgb_xgb_blend",
    },

    # Team records: the final features plus each team's past results.
    # Teams recur across battles; TeamRecordModel computes the records
    # inside each training fold.
    "team_blend": {
        "features": ["starter", "state_t30", "sacrifice",
                     "move_usage", "move_effects", "alive_species", "team_ids"],
        "model": "team_lgb_xgb_blend",
    },

    # Win conditions: team records plus, for each side, the Pokemon that the
    # other side's remaining team cannot stop, from a matchup chart learned
    # from the HP exchanges in the training battles.
    "wincon_blend": {
        "features": ["starter", "state_t30", "sacrifice", "move_usage",
                     "move_effects", "alive_species", "team_ids", "win_conditions"],
        "model": "team_lgb_xgb_blend",
    },
}
