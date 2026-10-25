"""
E04: where does the E01 model go wrong?

After two feature groups that did not help, we stop guessing and look at
the errors:

  1. Confidence. How many battles does the model call close to 50/50,
     and how accurate is it at each level of confidence?
  2. Slices. Accuracy by how far ahead P1 is at turn 30, and how often
     the side that is clearly ahead goes on to lose.
  3. Residuals. For every battle, residual = outcome - predicted win
     probability. A quantity that E01 does not use and that correlates
     with the residual carries information the model is missing. This
     lets us rank candidate features before building any of them.
  4. Whether the best candidates turn into CV accuracy.
  5. How the dataset was put together (knockouts on each side vs outcome).
  6. Two checks that the knockout counts in the data can be trusted.

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_species_table, find_data_dir, load_battles
from src.features import TEAM_SIZE, build_features, get_target, last_seen_states
from src.validation import N_SPLITS, SEED, cross_validate


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

    # 4. Do the candidates turn into accuracy? Same folds, both models.
    print("\n4. CV accuracy when the candidates are added to E01")
    behaviour = [c for c in C.columns if any(k in c for k in ("switches", "status_moves", "attacks"))]
    state = [c for c in C.columns if c not in behaviour]
    species_alive = alive_species_indicators(train)
    for name, extra in [("none (E01)", None), ("behaviour counts", C[behaviour]),
                        ("richer turn-30 state", C[state]), ("species still alive", species_alive)]:
        M = X if extra is None else pd.concat([X, extra], axis=1)
        print(f"   {name:22s} {both_models(M, y)}")

    # 5. How the data was put together: P1 win rate by knockouts on each side.
    print("\n5. P1 win rate by number of fainted Pokemon at turn 30 (rows P1, columns P2)")
    grid = pd.DataFrame({"p1": X["p1_n_fainted"], "p2": X["p2_n_fainted"], "won": y})
    print(grid.pivot_table(index="p1", columns="p2", values="won", aggfunc="mean").round(2).to_string())
    print(f"   Battles where P2 lost more than one Pokemon more than P1: "
          f"{(grid['p2'] > grid['p1'] + 1).sum()}")

    # 6. Two data quality checks behind the faint counts.
    print("\n6. Data quality")
    sacrifices = {"explosion", "selfdestruct"}
    for side in ("p1", "p2"):
        uses = [(t, i, b) for b in train for i, t in enumerate(b["battle_timeline"])
                if (t[f"{side}_move_details"] or {}).get("name") in sacrifices]
        marked = sum(t[f"{side}_pokemon_state"]["status"] == "fnt" for t, _, _ in uses)
        print(f"   {side}: Explosion/Self-Destruct used {len(uses)} times, marked as fainted {marked} times")
    # A knockout the timeline forgot to mark would look like a low-HP Pokemon
    # that leaves the field and never comes back. If those were real faints,
    # their count would explain part of the residual.
    for side in ("p1", "p2"):
        vanished = np.array([vanished_low_hp(b, side) for b in train])
        print(f"   {side}: low-HP Pokemon gone since turn 27 or earlier, mean {vanished.mean():.3f}, "
              f"residual corr {np.corrcoef(vanished, resid)[0, 1]:+.3f}")


def both_models(X, y):
    """CV accuracy of the main model and of the gradient boosting check."""
    lr = cross_validate("logreg_scaled", X, y)
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gb = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), X, y, cv=folds)
    return f"LR {lr['mean']:.4f} +/- {lr['std']:.4f}   GB {gb.mean():.4f} +/- {gb.std():.4f}"


def alive_species_indicators(battles, n_species=40):
    """For the 40 most common species: does each side still have it at turn 30?

    For P1 'has it' means it is in the team, for P2 that it was revealed.
    """
    counts = pd.Series([p["name"] for b in battles for p in b["p1_team_details"]]).value_counts()
    top = list(counts.index[:n_species])
    rows = []
    for b in battles:
        row = {}
        for side in ("p1", "p2"):
            states = last_seen_states(b, side)
            names = {p["name"] for p in b["p1_team_details"]} if side == "p1" else set(states)
            for n in top:
                row[f"{side}_alive_{n}"] = int(n in names and states.get(n, {}).get("status") != "fnt")
        rows.append(row)
    return pd.DataFrame(rows)


def vanished_low_hp(battle, side, max_hp=0.25, last_turn=27):
    """Pokemon last seen below 25% HP, not fainted, and absent since turn 27."""
    last, seen_at = {}, {}
    for t in battle["battle_timeline"]:
        s = t[f"{side}_pokemon_state"]
        last[s["name"]], seen_at[s["name"]] = s, t["turn"]
    return sum(1 for n, s in last.items()
               if s["status"] != "fnt" and s["hp_pct"] < max_hp and seen_at[n] <= last_turn)


if __name__ == "__main__":
    main()
