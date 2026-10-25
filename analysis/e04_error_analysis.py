"""
E04: where does the E01 model go wrong?

After two feature groups that did not help, we stop guessing and look at
the errors. Three parts:

  1. Confidence. How many battles does the model call close to 50/50,
     and how accurate is it at each level of confidence?
  2. Slices. Accuracy by how far ahead P1 is at turn 30, and how often
     the side that is clearly ahead goes on to lose.
  3. Residuals. For every battle, residual = outcome - predicted win
     probability. A quantity that E01 does not use and that correlates
     with the residual carries information the model is missing. This
     lets us rank candidate features before building any of them.

The candidates come from two places: a richer turn-30 state (what kind
of Pokemon are still standing, how healthy they are) and simple counts of
what the players did (switches, status moves). With 10,000 battles the
standard error of a correlation is about 0.01, so we treat |r| >= 0.03
as worth following up.

    python analysis/e04_error_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_species_table, find_data_dir, load_battles
from src.features import TEAM_SIZE, build_features, get_target, last_seen_states
from src.validation import cross_validate


# ---------------------------------------------------------------------------
# Candidate quantities, computed here only for the analysis. Whatever turns
# out to be useful gets a proper home in src/features.py afterwards.
# ---------------------------------------------------------------------------

def strength(name, species, default):
    """Base stat total of a species, scaled so an average OU Pokemon is ~1."""
    return species.get(name, {}).get("base_total", default) / default


def candidates(battle, species, avg_total):
    feats = {}
    timeline = battle["battle_timeline"]

    for side in ("p1", "p2"):
        states = last_seen_states(battle, side)
        alive = {n: s for n, s in states.items() if s["status"] != "fnt"}
        n_unseen = max(TEAM_SIZE - len(states), 0)

        # HP weighted by how strong each Pokemon is. Unseen Pokemon count
        # at full HP and average strength.
        feats[f"{side}_strength_hp"] = (
            sum(s["hp_pct"] * strength(n, species, avg_total) for n, s in alive.items()) + n_unseen
        )
        # Strongest Pokemon still standing (unseen ones count as average).
        alive_strengths = [strength(n, species, avg_total) for n in alive] + [1.0] * n_unseen
        feats[f"{side}_best_alive"] = max(alive_strengths, default=0)

        # How the remaining HP is spread: a Pokemon at 5% is nearly gone.
        feats[f"{side}_n_healthy"] = sum(s["hp_pct"] > 0.5 for s in alive.values()) + n_unseen
        feats[f"{side}_n_critical"] = sum(s["hp_pct"] < 0.25 for s in alive.values())

        # Sleep from Rest wears off after two turns, sleep from a move can last
        # much longer, so the two may not mean the same thing.
        resters = {t[f"{side}_pokemon_state"]["name"] for t in timeline
                   if (t[f"{side}_move_details"] or {}).get("name") == "rest"}
        sleeping = [n for n, s in alive.items() if s["status"] == "slp"]
        feats[f"{side}_n_rest_sleep"] = sum(n in resters for n in sleeping)
        feats[f"{side}_n_other_sleep"] = sum(n not in resters for n in sleeping)

        # Volatile effects on the active Pokemon at turn 30.
        effects = timeline[-1][f"{side}_pokemon_state"]["effects"]
        feats[f"{side}_active_effects"] = sum(e != "noeffect" for e in effects)

        # What the player did over the 30 turns.
        moves = [t[f"{side}_move_details"] for t in timeline]
        names = [t[f"{side}_pokemon_state"]["name"] for t in timeline]
        feats[f"{side}_n_switches"] = sum(a != b for a, b in zip(names, names[1:]))
        feats[f"{side}_n_status_moves"] = sum(m is not None and m["category"] == "status" for m in moves)
        feats[f"{side}_n_attacks"] = sum(m is not None and m["category"] != "status" for m in moves)

    for key in ("strength_hp", "best_alive", "n_healthy", "n_critical", "n_switches",
                "n_status_moves", "n_attacks"):
        feats[f"diff_{key}"] = feats[f"p1_{key}"] - feats[f"p2_{key}"]
    return feats


def main():
    train, _ = load_battles(find_data_dir())
    y = get_target(train)
    species = build_species_table(train)
    avg_total = np.mean([s["base_total"] for s in species.values()])

    # Out-of-fold predictions of E01, exactly as in the logged experiment.
    X = build_features(train, ["starter", "state_t30"]).drop(columns="battle_id")
    cv = cross_validate("logreg_scaled", X, y)
    p = cv["oof_proba"]
    correct = (p >= 0.5).astype(int) == y
    print(f"E01 out-of-fold accuracy: {correct.mean():.4f}\n")

    # 1. Confidence
    conf = np.abs(p - 0.5)
    bins = pd.cut(conf, [0, 0.05, 0.1, 0.2, 0.3, 0.5], include_lowest=True)
    table = pd.DataFrame({"bin": bins, "correct": correct}).groupby("bin", observed=True)["correct"]
    print("1. Accuracy by model confidence |p - 0.5|")
    print(pd.DataFrame({"share": table.size() / len(y), "accuracy": table.mean()}).round(3).to_string(), "\n")

    # 2. Slices by HP difference at turn 30
    d = X["diff_hp_total"]
    bins = pd.cut(d, [-6, -2, -1, -0.5, 0, 0.5, 1, 2, 6])
    g = pd.DataFrame({"bin": bins, "won": y, "correct": correct}).groupby("bin", observed=True)
    print("2. By team HP difference at turn 30 (P1 minus P2, in Pokemon units)")
    print(pd.DataFrame({"share": g.size() / len(y), "p1_win_rate": g["won"].mean(),
                        "accuracy": g["correct"].mean()}).round(3).to_string())
    upsets_p1 = ((p > 0.7) & (y == 0)).sum() / (p > 0.7).sum()
    upsets_p2 = ((p < 0.3) & (y == 1)).sum() / (p < 0.3).sum()
    print(f"   Model gives P1 > 70%: P1 still loses in {upsets_p1:.1%} of those battles")
    print(f"   Model gives P1 < 30%: P1 still wins in {upsets_p2:.1%} of those battles\n")

    # 3. Residual correlation with candidates
    C = pd.DataFrame([candidates(b, species, avg_total) for b in train])
    resid = y - p
    r = C.corrwith(pd.Series(resid)).sort_values(key=np.abs, ascending=False)
    print("3. Correlation of candidate quantities with the E01 residual")
    print(r.round(3).to_string())

    # Same, but only inside the battles the model finds hard (|p - 0.5| < 0.1).
    hard = conf < 0.1
    r_hard = C[hard].corrwith(pd.Series(resid[hard], index=C.index[hard]))
    print(f"\n   Inside the hard battles only ({hard.mean():.0%} of the data), top 8:")
    print(r_hard.sort_values(key=np.abs, ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
