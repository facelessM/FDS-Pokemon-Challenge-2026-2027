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
# E01: where the battle stands after turn 30
# ---------------------------------------------------------------------------

# Every team in this format has six Pokemon. We know all six for P1, but
# for P2 we only see the ones that have been sent out.
TEAM_SIZE = 6

# Non-volatile status conditions as they appear in the data. "fnt"
# (fainted) is handled separately since it means the Pokemon is out.
STATUSES = ["slp", "par", "psn", "tox", "frz", "brn"]


def last_seen_states(battle, side):
    """Return {pokemon name: its state the last time it was on the field}.

    The timeline only shows the active Pokemon on each turn. A Pokemon
    sitting on the bench keeps the HP and status it had when it left
    (Leftovers and Rest only work while active), so its last appearance
    is still the right picture of it at turn 30.
    """
    last = {}
    for turn in battle["battle_timeline"]:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state
    return last


def side_summary(states, n_unseen, prefix):
    """Summarise one side from its last-seen states.

    n_unseen is how many team members never showed up in the 30 turns.
    They have not taken any damage yet, so we count them at full HP.
    This holds for both sides: a Pokemon that has not switched in
    cannot have been hit (Spikes damage only happens on switch-in).
    """
    alive = [s for s in states.values() if s["status"] != "fnt"]
    n_fainted = len(states) - len(alive)

    feats = {
        f"{prefix}_n_fainted": n_fainted,
        f"{prefix}_n_alive": TEAM_SIZE - n_fainted,
        f"{prefix}_n_seen": len(states),
        # Total remaining HP across the team, in "Pokemon" units: 6.0 means
        # the whole team is untouched, 0 means everyone has fainted.
        f"{prefix}_hp_total": sum(s["hp_pct"] for s in alive) + n_unseen,
    }

    # How many living team members carry each status.
    for status in STATUSES:
        feats[f"{prefix}_n_{status}"] = sum(s["status"] == status for s in alive)
    feats[f"{prefix}_n_statused"] = sum(s["status"] != "nostatus" for s in alive)

    return feats


def active_summary(state, prefix):
    """The Pokemon on the field at turn 30: HP, status and stat boosts."""
    return {
        f"{prefix}_active_hp": state["hp_pct"],
        f"{prefix}_active_statused": int(state["status"] not in ("nostatus", "fnt")),
        f"{prefix}_active_boosts": sum(state["boosts"].values()),
    }


def state_t30_features(battle):
    """Snapshot of the battle at the end of the visible part (turn 30).

    For each side: how many Pokemon fainted, how much HP the team has
    left, how many are statused, and the condition of the active one.
    Then the P1 minus P2 differences, since who is ahead matters more
    than the raw numbers.
    """
    p1_states = last_seen_states(battle, "p1")
    p2_states = last_seen_states(battle, "p2")

    # P1: we know the full roster, so unseen = team members that never came in.
    p1_names = {p["name"] for p in battle["p1_team_details"]}
    p1_unseen = len(p1_names - set(p1_states))
    # P2: we do not know the roster, so we assume a full team of six.
    p2_unseen = max(TEAM_SIZE - len(p2_states), 0)

    feats = {}
    feats.update(side_summary(p1_states, p1_unseen, "p1"))
    feats.update(side_summary(p2_states, p2_unseen, "p2"))

    last_turn = battle["battle_timeline"][-1]
    feats.update(active_summary(last_turn["p1_pokemon_state"], "p1"))
    feats.update(active_summary(last_turn["p2_pokemon_state"], "p2"))

    # Positive values mean P1 is ahead.
    feats["diff_n_alive"] = feats["p1_n_alive"] - feats["p2_n_alive"]
    feats["diff_hp_total"] = feats["p1_hp_total"] - feats["p2_hp_total"]
    feats["diff_n_statused"] = feats["p2_n_statused"] - feats["p1_n_statused"]
    feats["diff_active_hp"] = feats["p1_active_hp"] - feats["p2_active_hp"]
    feats["diff_active_boosts"] = feats["p1_active_boosts"] - feats["p2_active_boosts"]

    return feats


# ---------------------------------------------------------------------------
# E02: how the battle got to turn 30
# ---------------------------------------------------------------------------

# Turns at which we take a reading of the HP difference. Turn 30 itself is
# already covered by diff_hp_total in E01, so we stop at 25.
CHECKPOINTS = [5, 10, 15, 20, 25]

# Value used for "first faint" when a side lost nothing in 30 turns.
# One past the last turn, so "never" still sorts after "late".
NO_FAINT = 31


def team_series(battle, side, n_known):
    """Team HP and faint count for one side, after every turn.

    Same idea as last_seen_states, but we take a reading at each turn
    instead of only at the end. n_known is the roster size we assume
    (6 for both sides), and every Pokemon not seen yet counts at full HP.

    Returns two lists of length 30: team HP and number fainted.
    """
    last = {}
    hp_series, faint_series = [], []
    for turn in battle["battle_timeline"]:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state

        alive = [s for s in last.values() if s["status"] != "fnt"]
        n_unseen = max(n_known - len(last), 0)
        hp_series.append(sum(s["hp_pct"] for s in alive) + n_unseen)
        faint_series.append(len(last) - len(alive))
    return hp_series, faint_series


def count_lead_changes(diff):
    """How many times the side in front changed.

    Turns where the two teams are exactly level are skipped, so a run
    like +, 0, + does not count as two changes.
    """
    signs = [np.sign(d) for d in diff if d != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def trajectory_features(battle):
    """Summaries of the 30-turn HP race between the two teams.

    E01 tells us where the battle ended up at turn 30. Here we look at the
    path: was P1 ahead the whole time, did it fall behind and recover, is
    the gap growing or shrinking at the end? If momentum carries over into
    the hidden turns, these should add something on top of the snapshot.
    """
    p1_hp, p1_faints = team_series(battle, "p1", TEAM_SIZE)
    p2_hp, p2_faints = team_series(battle, "p2", TEAM_SIZE)

    # HP difference after each turn, positive when P1 is ahead.
    diff = np.array(p1_hp) - np.array(p2_hp)
    turns = np.arange(1, len(diff) + 1)

    feats = {}

    # Level of the race at a few points along the way.
    for t in CHECKPOINTS:
        feats[f"diff_hp_t{t}"] = diff[t - 1]

    # Overall shape of the race.
    feats["diff_hp_mean"] = diff.mean()
    feats["diff_hp_min"] = diff.min()
    feats["diff_hp_max"] = diff.max()
    # Slope of a straight line through the 30 points, in HP per turn.
    feats["diff_hp_slope"] = np.polyfit(turns, diff, 1)[0]
    feats["turns_p1_ahead"] = int((diff > 0).sum())
    feats["n_lead_changes"] = count_lead_changes(diff)

    # Momentum at the very end of the visible window.
    feats["diff_hp_change_last5"] = diff[-1] - diff[-6]
    feats["diff_hp_change_last10"] = diff[-1] - diff[-11]
    feats["p1_hp_change_last5"] = p1_hp[-1] - p1_hp[-6]
    feats["p2_hp_change_last5"] = p2_hp[-1] - p2_hp[-6]

    # When the knockouts happened. A team that lost its first Pokemon at
    # turn 5 is in a different spot from one that lost it at turn 29.
    feats["p1_first_faint_turn"] = next((t for t, f in zip(turns, p1_faints) if f > 0), NO_FAINT)
    feats["p2_first_faint_turn"] = next((t for t, f in zip(turns, p2_faints) if f > 0), NO_FAINT)
    feats["p1_faints_last10"] = p1_faints[-1] - p1_faints[-11]
    feats["p2_faints_last10"] = p2_faints[-1] - p2_faints[-11]

    return feats


# ---------------------------------------------------------------------------
# Registry: name used in experiment configs -> function
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "starter": starter_features,
    "state_t30": state_t30_features,
    "trajectory": trajectory_features,
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
