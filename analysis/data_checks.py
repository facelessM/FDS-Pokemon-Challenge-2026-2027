"""
How the timelines record knockouts.

In E04 we read the faint counts at turn 30 as they are and concluded that P1
is almost always level or behind. This script shows why that was wrong: when
P1 knocks out a P2 Pokemon with an attack, the timeline usually does not mark
it as fainted. The Pokemon just leaves the field and is never seen again.

  1. Low-HP Pokemon hit by an attack: what happens on the next row, per side.
  2. P2 faints that are recorded: how many happen on the same row as a P1 faint
     (Explosion and Self-Destruct trades).
  3. Whether counting the hidden knockouts helps the final model (CV seeds 1-5).

    python analysis/data_checks.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.validation import cross_validate

LOW_HP = 0.3


def is_attack(move):
    return bool(move) and move["category"] != "status"


def after_low_hp_hit(battles, side):
    """For a side's Pokemon below LOW_HP that take an attack, count what the
    next row shows: fainted, still in, or gone (and whether it ever comes back)."""
    opp = "p2" if side == "p1" else "p1"
    out = Counter()
    for b in battles:
        tl = b["battle_timeline"]
        for i in range(1, len(tl)):
            prev, cur = tl[i - 1][f"{side}_pokemon_state"], tl[i][f"{side}_pokemon_state"]
            if prev["status"] == "fnt" or prev["hp_pct"] >= LOW_HP or not is_attack(tl[i][f"{opp}_move_details"]):
                continue
            if cur["name"] == prev["name"]:
                out["fainted" if cur["status"] == "fnt" else "still in"] += 1
            elif any(t[f"{side}_pokemon_state"]["name"] == prev["name"] for t in tl[i:]):
                out["gone, comes back"] += 1
            else:
                out["gone, never seen again"] += 1
    return out


def hidden_kos(b):
    """P2 Pokemon below LOW_HP that left the field right after a P1 attack and
    never appeared again, while not marked as fainted."""
    tl = b["battle_timeline"]
    last_row = {t["p2_pokemon_state"]["name"]: i for i, t in enumerate(tl)}
    names = set()
    for i in range(1, len(tl)):
        prev, cur = tl[i - 1]["p2_pokemon_state"], tl[i]["p2_pokemon_state"]
        if (prev["name"] != cur["name"] and prev["status"] != "fnt" and prev["hp_pct"] < LOW_HP
                and last_row[prev["name"]] == i - 1 and is_attack(tl[i]["p1_move_details"])):
            names.add(prev["name"])
    return names


def corrected_counts(b):
    """P2 faints and remaining HP at turn 30 with the hidden knockouts added."""
    last = {t["p2_pokemon_state"]["name"]: t["p2_pokemon_state"] for t in b["battle_timeline"]}
    hidden = hidden_kos(b)
    alive = [n for n, s in last.items() if s["status"] != "fnt" and n not in hidden]
    return {"p2_hidden_ko": len(hidden),
            "p2_fainted_corrected": len(last) - len(alive),
            "p2_hp_corrected": sum(last[n]["hp_pct"] for n in alive) + max(6 - len(last), 0)}


def main():
    data_dir = find_data_dir()
    train, _ = load_battles(data_dir)
    y = get_target(train)

    print(f"1. Pokemon below {LOW_HP:.0%} HP hit by an attack, next row")
    for side in ("p1", "p2"):
        c = after_low_hp_hit(train, side)
        n = sum(c.values())
        shares = ", ".join(f"{k} {v / n:.1%}" for k, v in c.items())
        print(f"   {side.upper()} (n={n}): {shares}")
        print(f"      fainted + never seen again: {c['fainted'] + c['gone, never seen again']}")

    same_row = total = 0
    for b in train:
        for t in b["battle_timeline"]:
            if t["p2_pokemon_state"]["status"] == "fnt":
                total += 1
                same_row += t["p1_pokemon_state"]["status"] == "fnt"
    print(f"\n2. Rows with a fainted P2 Pokemon: {total}, of which {same_row / total:.1%} "
          f"also show a fainted P1 Pokemon")

    context = {
        "type_chart": load_type_chart(data_dir),
        "species": build_species_table(train),
        "move_vocab": build_move_vocab(train),
        "top_species": build_top_species(train),
    }
    X = build_features(train, EXPERIMENTS["final_blend"]["features"], context).drop(columns="battle_id")
    C = pd.DataFrame([corrected_counts(b) for b in train])
    print(f"\n3. Hidden P2 knockouts per battle: {C.p2_hidden_ko.mean():.2f}")
    for label, M in [("final features", X), ("+ corrected counts", pd.concat([X, C], axis=1))]:
        s = [cross_validate("lgbm_final", M, y, seed=k)["mean"] for k in (1, 2, 3, 4, 5)]
        print(f"   LightGBM, {label:20s} seeds 1-5: {np.mean(s):.4f}  ({', '.join(f'{v:.4f}' for v in s)})")


if __name__ == "__main__":
    main()
