"""
How much of the final CV score is real?

The usual folds (seed 42) were also used while we developed the features,
so a score on them is slightly optimistic. Here the final model, LightGBM
alone and the E05 baseline are scored on CV seeds 6 to 10, which were never
used during development (seeds 1 to 5 were used to compare the final
candidates).

    python analysis/final_checks.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.validation import cross_validate

SEEDS = [6, 7, 8, 9, 10]


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

    # (label, experiment, model). The third row is LightGBM alone on the
    # final features, to see what the blend adds.
    runs = [("e05_sacrifice", "e05_sacrifice", None),
            ("final_blend", "final_blend", None),
            ("lightgbm only", "final_blend", "lgbm_final")]
    for label, exp, model in runs:
        config = EXPERIMENTS[exp]
        X = build_features(train, config["features"], context).drop(columns="battle_id")
        scores = [cross_validate(model or config["model"], X, y, seed=s)["mean"] for s in SEEDS]
        print(f"{label:14s} seeds 6-10: {np.mean(scores):.4f} +/- {np.std(scores):.4f}  "
              f"({', '.join(f'{s:.4f}' for s in scores)})")


if __name__ == "__main__":
    main()
