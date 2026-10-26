"""
What the final model learns, in game terms.

Adding Gen 2 mechanics as explicit features did not change the score: the
model already picks them up from turn-30 HP, move usage and move effects.
These tables show the mechanics directly on the training data.

  1. Spikes left on a side at turn 30 with no way to remove them (no Rapid
     Spin user standing, or a Ghost on the other side to block it).
  2. How well each side's attacks matched their targets' types.
  3. Final model accuracy by the number of Rest users in the battle.

    python analysis/mechanics_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (build_move_vocab, build_species_table, build_top_species,
                      find_data_dir, load_battles, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target, last_seen_states, type_multiplier
from src.validation import cross_validate


def side_types(battle, side, species):
    """Types of every Pokemon we know for a side (P1: team, P2: revealed)."""
    if side == "p1":
        return {p["name"]: [t for t in p["types"] if t != "notype"] for p in battle["p1_team_details"]}
    names = {t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"]}
    return {n: species.get(n, {"types": []})["types"] for n in names}


def spikes_stuck(battle, species):
    """1 if Spikes are on that side at turn 30 and it cannot clear them."""
    timeline = battle["battle_timeline"]
    on_side = {"p1": 0, "p2": 0}
    spinners = {"p1": set(), "p2": set()}
    for turn in timeline:
        for me, opp in (("p1", "p2"), ("p2", "p1")):
            move = (turn[f"{me}_move_details"] or {}).get("name")
            if move == "spikes":
                on_side[opp] = 1
            if move == "rapidspin":
                on_side[me] = 0
                spinners[me].add(turn[f"{me}_pokemon_state"]["name"])
    out = {}
    for me, opp in (("p1", "p2"), ("p2", "p1")):
        alive_me = {n for n, s in last_seen_states(battle, me).items() if s["status"] != "fnt"}
        opp_states = last_seen_states(battle, opp)
        opp_types = side_types(battle, opp, species)
        ghost_blocker = any("ghost" in opp_types.get(n, []) for n, s in opp_states.items()
                            if s["status"] != "fnt")
        can_spin = bool(spinners[me] & alive_me) and not ghost_blocker
        out[me] = int(on_side[me] == 1 and not can_spin)
    return out["p1"] - out["p2"]


def attack_effectiveness(battle, side, species, chart):
    """Mean type multiplier of the side's attacks against the Pokemon they hit."""
    opp = "p2" if side == "p1" else "p1"
    opp_types = side_types(battle, opp, species)
    mults = [type_multiplier(t[f"{side}_move_details"]["type"],
                             opp_types.get(t[f"{opp}_pokemon_state"]["name"], []), chart)
             for t in battle["battle_timeline"]
             if t[f"{side}_move_details"] and t[f"{side}_move_details"]["category"] != "status"]
    return np.mean(mults) if mults else 1.0


def main():
    data_dir = find_data_dir()
    train, _ = load_battles(data_dir)
    y = pd.Series(get_target(train))
    chart, species = load_type_chart(data_dir), build_species_table(train)

    stuck = pd.Series([spikes_stuck(b, species) for b in train])
    print("1. P1 win rate by Spikes stuck at turn 30 (P1 side minus P2 side)")
    print(y.groupby(stuck).agg(["mean", "size"]).round(3).to_string(), "\n")

    eff = pd.Series([attack_effectiveness(b, "p1", species, chart) - attack_effectiveness(b, "p2", species, chart)
                     for b in train])
    quintile = pd.qcut(eff.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    print("2. P1 win rate by quintile of attack effectiveness (P1 minus P2)")
    print(y.groupby(quintile, observed=True).mean().round(3).to_string(), "\n")

    context = {"type_chart": chart, "species": species,
               "move_vocab": build_move_vocab(train), "top_species": build_top_species(train)}
    X = build_features(train, EXPERIMENTS["final_blend"]["features"], context).drop(columns="battle_id")
    correct = (cross_validate("lgbm_final", X, y.values)["oof_proba"] >= 0.5) == y.values
    resters = pd.Series([
        sum(any((t[f"{s}_move_details"] or {}).get("name") == "rest" and t[f"{s}_pokemon_state"]["name"] == n
                for t in b["battle_timeline"])
            for s in ("p1", "p2") for n in last_seen_states(b, s))
        for b in train]).clip(upper=4)
    print("3. Final model accuracy (out of fold) by number of Rest users in the battle")
    print(pd.Series(correct).groupby(resters).agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
