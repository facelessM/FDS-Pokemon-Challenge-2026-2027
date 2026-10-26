"""
Teams recur across battles, and their past results predict the next one.

  1. How often teams come back, in train and test, and how far apart the
     win rates of the most played teams are.
  2. How often P2's team can be matched from the Pokemon it shows.
  3. The final model with and without team records, on CV seeds 6 to 10.

    python analysis/team_checks.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (build_move_vocab, build_species_table, build_team_index,
                      build_top_species, find_data_dir, load_battles,
                      load_type_chart, team_key)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target, team_id_features
from src.validation import cross_validate

SEEDS = [6, 7, 8, 9, 10]


def main():
    data_dir = find_data_dir()
    train, test = load_battles(data_dir)
    y = get_target(train)
    context = {
        "type_chart": load_type_chart(data_dir),
        "species": build_species_table(train),
        "move_vocab": build_move_vocab(train),
        "top_species": build_top_species(train),
        "teams": build_team_index(train),
    }

    counts = Counter(team_key(b) for b in train)
    print(f"1. {len(train)} training battles, {len(counts)} distinct P1 teams")
    print(f"   battles whose P1 team plays more than once: {np.mean([counts[team_key(b)] > 1 for b in train]):.1%}")
    print(f"   test battles whose P1 team is in the training set: {np.mean([team_key(b) in counts for b in test]):.1%}")
    results = defaultdict(list)
    for b, won in zip(train, y):
        results[team_key(b)].append(won)
    rates = [np.mean(r) for r in results.values() if len(r) >= 30]
    print(f"   teams with 30+ battles: {len(rates)}, P1 win rate from {min(rates):.0%} to {max(rates):.0%}")

    ids = [team_id_features(b, context) for b in train]
    cand = np.array([i["p2_team_candidates"] for i in ids])
    print(f"\n2. P2 Pokemon seen fit a known team: {np.mean(cand > 0):.1%}, a single team: {np.mean(cand == 1):.1%}")

    print("\n3. CV seeds 6 to 10")
    for exp in ("final_blend", "team_blend"):
        config = EXPERIMENTS[exp]
        X = build_features(train, config["features"], context).drop(columns="battle_id")
        scores = [cross_validate(config["model"], X, y, seed=s)["mean"] for s in SEEDS]
        print(f"   {exp:12s} {np.mean(scores):.4f} +/- {np.std(scores):.4f}  ({', '.join(f'{s:.4f}' for s in scores)})")


if __name__ == "__main__":
    main()
