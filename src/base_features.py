"""State, move observations and species features used by the four core models."""

from collections import Counter
import numpy as np
import pandas as pd

TEAM_SIZE = 6
STATUSES = ["slp", "par", "psn", "tox", "frz", "brn"]
SACRIFICE_MOVES = {"explosion", "selfdestruct"}


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


def sacrifice_features(battle):
    """Split each side's fainted count into sacrifices and knockouts taken.

    E01 counts every fainted Pokemon the same way. But a Pokemon that used
    Explosion chose to go down, usually taking a big chunk out of the
    opponent, which is not the same as being knocked out. The error
    analysis (E04) showed P1 does this far more often than P2.
    """
    feats = {}
    for side in ("p1", "p2"):
        n_fainted = sum(
            s["status"] == "fnt" for s in last_seen_states(battle, side).values()
        )
        n_sacrifice = sum(
            (t[f"{side}_move_details"] or {}).get("name") in SACRIFICE_MOVES
            for t in battle["battle_timeline"]
        )
        feats[f"{side}_n_sacrifice"] = n_sacrifice
        feats[f"{side}_n_ko_taken"] = n_fainted - n_sacrifice

    feats["diff_sacrifice"] = feats["p1_n_sacrifice"] - feats["p2_n_sacrifice"]
    feats["diff_ko_taken"] = feats["p1_n_ko_taken"] - feats["p2_n_ko_taken"]
    return feats


def move_usage_features(battle, context):
    """How many times each side used each move in the 30 turns.

    The turn-30 state says who is ahead. The moves say how each side plays:
    stalling with Rest and Toxic, setting Spikes, going for Explosion trades.
    The list of moves comes from the training battles (context["move_vocab"]).
    """
    feats = {}
    for side in ("p1", "p2"):
        used = Counter(
            t[f"{side}_move_details"]["name"]
            for t in battle["battle_timeline"]
            if t[f"{side}_move_details"]
        )
        for move in context["move_vocab"][side]:
            feats[f"{side}_mv_{move}"] = used.get(move, 0)
    return feats


def move_effect_features(battle):
    """What the moves actually did, read from HP changes between turns.

    For each side: damage dealt per attacking move to the opponent's active
    Pokemon (only when the same Pokemon is on the field before and after),
    attacks that did nothing, HP healed, turns lost to sleep, paralysis or
    freeze, and turns spent asleep or paralysed.
    """
    timeline = battle["battle_timeline"]
    feats = {}
    for me, opp in (("p1", "p2"), ("p2", "p1")):
        damage, healed, lost, asleep, paralysed, no_effect = [], 0.0, 0, 0, 0, 0
        for prev, cur in zip(timeline, timeline[1:]):
            move = cur[f"{me}_move_details"]
            before, after = prev[f"{opp}_pokemon_state"], cur[f"{opp}_pokemon_state"]
            if (
                move
                and move["category"] != "status"
                and before["name"] == after["name"]
            ):
                hit = before["hp_pct"] - after["hp_pct"]
                damage.append(hit)
                no_effect += hit <= 0

            mine_before, mine = prev[f"{me}_pokemon_state"], cur[f"{me}_pokemon_state"]
            same = mine_before["name"] == mine["name"]
            if same and mine["hp_pct"] > mine_before["hp_pct"]:
                healed += mine["hp_pct"] - mine_before["hp_pct"]
            if move is None and same and mine["status"] in ("slp", "par", "frz"):
                lost += 1
            asleep += mine["status"] == "slp"
            paralysed += mine["status"] == "par"

        damage = np.asarray(damage, dtype=float)
        feats[f"{me}_dmg_mean"] = damage.mean() if damage.size else 0.0
        feats[f"{me}_dmg_total"] = damage.clip(min=0).sum()
        feats[f"{me}_dmg_max"] = damage.max() if damage.size else 0.0
        feats[f"{me}_n_hits"] = len(damage)
        feats[f"{me}_no_effect_hits"] = no_effect
        feats[f"{me}_heal_total"] = healed
        feats[f"{me}_turns_lost"] = lost
        feats[f"{me}_turns_slp"] = asleep
        feats[f"{me}_turns_par"] = paralysed

    for key in (
        "dmg_mean",
        "dmg_total",
        "dmg_max",
        "heal_total",
        "turns_lost",
        "turns_slp",
        "turns_par",
        "no_effect_hits",
    ):
        feats[f"diffd_{key}"] = feats[f"p1_{key}"] - feats[f"p2_{key}"]
    return feats


def alive_species_features(battle, context):
    """For the most common species: does each side still have it at turn 30?

    For P1 that means it is on the team and has not fainted (a Pokemon that
    never came in counts as alive). For P2 it means it was revealed and has
    not fainted. The species list comes from the training battles.
    """
    feats = {}
    for side in ("p1", "p2"):
        states = last_seen_states(battle, side)
        if side == "p1":
            roster = {p["name"] for p in battle["p1_team_details"]}
        else:
            roster = set(states)
        for name in context["top_species"]:
            fainted = states.get(name, {}).get("status") == "fnt"
            feats[f"{side}_alive_{name}"] = int(name in roster and not fainted)
    return feats


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


FEATURE_GROUPS = {
    "starter": starter_features,
    "state_t30": state_t30_features,
    "sacrifice": sacrifice_features,
    "move_usage": move_usage_features,
    "move_effects": move_effect_features,
    "alive_species": alive_species_features,
}
