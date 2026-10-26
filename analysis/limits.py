"""
Where the final model goes wrong, and why.

  1. P1's win rate by the Pokemon lead at turn 30 (hidden P2 knockouts counted,
     see data_checks.py), with the final model's accuracy in each group.
  2. The final model's accuracy by how sure it is.

Out-of-fold predictions of the final model on the usual folds (seed 42).

    python analysis/limits.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from data_checks import corrected_counts
from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.validation import cross_validate


def p1_fainted(b):
    last = {t["p1_pokemon_state"]["name"]: t["p1_pokemon_state"] for t in b["battle_timeline"]}
    return sum(s["status"] == "fnt" for s in last.values())


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
    X = build_features(train, EXPERIMENTS["final_blend"]["features"], context).drop(columns="battle_id")
    p = cross_validate("lgb_xgb_blend", X, y)["oof_proba"]

    lead = np.array([corrected_counts(b)["p2_fainted_corrected"] - p1_fainted(b) for b in train])
    d = pd.DataFrame({"won": y, "right": (p >= 0.5) == y, "lead": np.clip(lead, -3, 2)})
    g = d.groupby("lead")
    print("1. P1 Pokemon lead at turn 30 (-3 = three or more down, 2 = two or more up)")
    print(pd.DataFrame({"battles": g.size(), "P1 wins": g.won.mean().round(3),
                        "model accuracy": g.right.mean().round(3)}).to_string())

    d["distance"] = pd.cut(np.abs(p - 0.5), [0, 0.1, 0.2, 0.3, 0.5], include_lowest=True,
                           labels=["< 10 points", "10-20", "20-30", "> 30"])
    g = d.groupby("distance", observed=True)
    print("\n2. Accuracy by distance of the predicted probability from 50/50")
    print(pd.DataFrame({"share": (g.size() / len(d)).round(3), "accuracy": g.right.mean().round(3)}).to_string())


if __name__ == "__main__":
    main()
