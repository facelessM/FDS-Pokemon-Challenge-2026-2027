"""
Feature engineering.

Each feature group is a small function that takes one battle (a dict)
and returns a dict of numbers. Groups are registered in FEATURE_GROUPS,
and an experiment just lists the groups it wants. This way adding a new
idea means writing one function and adding one line to the registry,
and older experiments keep working exactly as before.

Rule we follow everywhere in this file: a feature can only use what is
inside the battle itself (plus the supplied type chart). No outside
tables, no usage stats, nothing typed in by hand about specific Pokemon.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# E0: the starter notebook features, copied as they are
# ---------------------------------------------------------------------------

def starter_features(battle):
    """Same six features as the official starter notebook.

    We keep this group untouched on purpose so that E0 reproduces the
    starter score and gives us a fair point of comparison.
    """
    feats = {}

    p1_team = battle.get("p1_team_details", [])
    if p1_team:
        feats["p1_mean_hp"] = np.mean([p.get("base_hp", 0) for p in p1_team])
        feats["p1_mean_spe"] = np.mean([p.get("base_spe", 0) for p in p1_team])
        feats["p1_mean_atk"] = np.mean([p.get("base_atk", 0) for p in p1_team])
        feats["p1_mean_def"] = np.mean([p.get("base_def", 0) for p in p1_team])

    p2_lead = battle.get("p2_lead_details")
    if p2_lead:
        feats["p2_lead_hp"] = p2_lead.get("base_hp", 0)
        feats["p2_lead_spe"] = p2_lead.get("base_spe", 0)

    return feats


# ---------------------------------------------------------------------------
# Registry: name used in experiment configs -> function
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "starter": starter_features,
}


def build_features(battles, groups, context=None):
    """Turn a list of battles into a DataFrame.

    battles : list of battle dicts
    groups  : list of group names from FEATURE_GROUPS
    context : optional dict with shared resources (for example the type
              chart). Groups that need it accept it as a second argument.

    The result always has battle_id as its first column. Missing values
    are filled with 0, same as the starter notebook.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise KeyError(f"Unknown feature group(s): {unknown}")

    rows = []
    for battle in battles:
        row = {"battle_id": battle["battle_id"]}
        for name in groups:
            func = FEATURE_GROUPS[name]
            # Only pass the context to groups that actually ask for it.
            if func.__code__.co_argcount > 1:
                row.update(func(battle, context))
            else:
                row.update(func(battle))
        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


def get_target(battles):
    """Pull player_won out of the training battles as a 0/1 numpy array."""
    return np.array([int(b["player_won"]) for b in battles])
