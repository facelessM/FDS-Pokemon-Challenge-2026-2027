"""
Win conditions: does a side still have a Pokemon the other cannot stop?

  1. Size of the matchup chart learned from the HP exchanges.
  2. P1's win rate by the difference in unanswered Pokemon at turn 30.
  3. Team records with and without win conditions, on CV seeds 6 to 10.

    python analysis/wincon_checks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (build_matchup_chart, build_move_vocab, build_species_table,
                      build_team_index, build_top_species, find_data_dir,
                      load_battles, load_type_chart)
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
        "teams": build_team_index(train),
        "matchups": build_matchup_chart(train),
    }
    turns = context["matchups"]["turns"]
    print(f"1. Species pairs in the chart: {len(turns) // 2}, "
          f"with 20+ turns of exchanges: {sum(t >= 20 for t in turns.values()) // 2}")

    W = build_features(train, ["team_ids", "win_conditions"], context)
    d = pd.DataFrame({"won": y, "diff": W["diff_unanswered"].clip(-3, 3)})
    g = d.groupby("diff")
    print("\n2. P1 unanswered minus P2 unanswered at turn 30 (clipped at +/-3)")
    print(pd.DataFrame({"battles": g.size(), "P1 wins": g.won.mean().round(3)}).to_string())

    print("\n3. CV seeds 6 to 10")
    for exp in ("team_blend", "wincon_blend"):
        config = EXPERIMENTS[exp]
        X = build_features(train, config["features"], context).drop(columns="battle_id")
        scores = [cross_validate(config["model"], X, y, seed=s)["mean"] for s in SEEDS]
        print(f"   {exp:13s} {np.mean(scores):.4f} +/- {np.std(scores):.4f}  ({', '.join(f'{s:.4f}' for s in scores)})")


if __name__ == "__main__":
    main()
