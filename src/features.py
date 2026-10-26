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

from collections import Counter, defaultdict

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
# E03: how the two teams match up
# ---------------------------------------------------------------------------

def type_multiplier(attack_type, defender_types, chart):
    """Damage multiplier of one attacking type against a Pokemon.

    Dual types multiply: Ground against Fire/Flying is 2 x 0 = 0.
    The chart only lists non-neutral pairs, so anything missing is 1.
    """
    mult = 1.0
    for d in defender_types:
        mult *= chart.get((attack_type, d), 1.0)
    return mult


def best_stab(attacker_types, defender_types, chart):
    """Best multiplier the attacker gets from its own types.

    Pokemon get a bonus for moves that share their type (STAB), so their
    own types are the attacks they are most likely to carry. We do not
    know the full movesets, so this is the closest thing we can compute
    before the battle starts.
    """
    if not attacker_types or not defender_types:
        return 1.0
    return max(type_multiplier(a, defender_types, chart) for a in attacker_types)


def matchup_features(battle, context):
    """Relative strength of the two teams: base stats and type matchups.

    The starter features looked at P1's team on its own, and they told us
    nothing because most OU teams look alike. Here every number compares
    P1 with what we know of P2: its lead, the Pokemon it revealed during
    the 30 turns, and the two Pokemon on the field at the end.
    """
    chart, species = context["type_chart"], context["species"]

    p1_team = [
        {"types": [t for t in p["types"] if t != "notype"],
         "base_total": sum(p[f"base_{s}"] for s in ("hp", "atk", "def", "spa", "spd", "spe")),
         "base_spe": p["base_spe"]}
        for p in battle["p1_team_details"]
    ]
    p2_lead_types = [t for t in battle["p2_lead_details"]["types"] if t != "notype"]

    # P2 Pokemon revealed in the timeline, looked up in the species table.
    # The very few species missing from the table are left out.
    p2_names = {t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"]}
    p2_seen = [species[n] for n in p2_names if n in species]

    feats = {}

    # Raw strength. Base stat total is the usual single number for how
    # strong a species is. For P2 we can only use the revealed ones.
    p1_totals = [p["base_total"] for p in p1_team]
    p2_totals = [p["base_total"] for p in p2_seen] or [0]
    feats["p1_base_total_mean"] = np.mean(p1_totals)
    feats["p2_seen_base_total_mean"] = np.mean(p2_totals)
    feats["diff_base_total_mean"] = feats["p1_base_total_mean"] - feats["p2_seen_base_total_mean"]
    feats["diff_speed_max"] = max(p["base_spe"] for p in p1_team) - max((p["base_spe"] for p in p2_seen), default=0)

    # Team build. Low levels and a missing Leftovers are rare in OU and
    # often point to an unusual or joke team.
    feats["p1_n_types"] = len({t for p in p1_team for t in p["types"]})
    feats["p1_n_low_level"] = sum(p["level"] < 100 for p in battle["p1_team_details"])
    feats["p1_n_leftovers"] = sum(p.get("item") == "leftovers" for p in battle["p1_team_details"])

    # Against P2's lead: how many P1 Pokemon hit it super effectively with
    # their own types, and how many it hits super effectively back.
    p1_vs_lead = [best_stab(p["types"], p2_lead_types, chart) for p in p1_team]
    lead_vs_p1 = [best_stab(p2_lead_types, p["types"], chart) for p in p1_team]
    feats["p1_n_se_vs_lead"] = sum(m > 1 for m in p1_vs_lead)
    feats["lead_n_se_vs_p1"] = sum(m > 1 for m in lead_vs_p1)
    feats["diff_se_vs_lead"] = feats["p1_n_se_vs_lead"] - feats["lead_n_se_vs_p1"]

    # Team against team, using every P2 Pokemon we have seen. For each pair
    # we take the best type multiplier and average over all pairs.
    if p2_seen:
        p1_off = [best_stab(a["types"], d["types"], chart) for a in p1_team for d in p2_seen]
        p2_off = [best_stab(a["types"], d["types"], chart) for a in p2_seen for d in p1_team]
        feats["p1_type_offense"] = np.mean(p1_off)
        feats["p2_type_offense"] = np.mean(p2_off)
    else:
        feats["p1_type_offense"] = feats["p2_type_offense"] = 1.0
    feats["diff_type_offense"] = feats["p1_type_offense"] - feats["p2_type_offense"]

    # The two Pokemon on the field at turn 30: who has the type edge right now?
    last = battle["battle_timeline"][-1]
    p1_active = species.get(last["p1_pokemon_state"]["name"], {"types": []})
    p2_active = species.get(last["p2_pokemon_state"]["name"], {"types": []})
    feats["active_p1_offense"] = best_stab(p1_active["types"], p2_active["types"], chart)
    feats["active_p2_offense"] = best_stab(p2_active["types"], p1_active["types"], chart)
    feats["diff_active_offense"] = feats["active_p1_offense"] - feats["active_p2_offense"]

    return feats


# ---------------------------------------------------------------------------
# E05: sacrifices are not knockouts
# ---------------------------------------------------------------------------

# Moves that make the user faint on purpose to deal a big hit.
SACRIFICE_MOVES = {"explosion", "selfdestruct"}


def sacrifice_features(battle):
    """Split each side's fainted count into sacrifices and knockouts taken.

    E01 counts every fainted Pokemon the same way. But a Pokemon that used
    Explosion chose to go down, usually taking a big chunk out of the
    opponent, which is not the same as being knocked out. The error
    analysis (E04) showed P1 does this far more often than P2.
    """
    feats = {}
    for side in ("p1", "p2"):
        n_fainted = sum(s["status"] == "fnt" for s in last_seen_states(battle, side).values())
        n_sacrifice = sum(
            (t[f"{side}_move_details"] or {}).get("name") in SACRIFICE_MOVES
            for t in battle["battle_timeline"]
        )
        feats[f"{side}_n_sacrifice"] = n_sacrifice
        feats[f"{side}_n_ko_taken"] = n_fainted - n_sacrifice

    feats["diff_sacrifice"] = feats["p1_n_sacrifice"] - feats["p2_n_sacrifice"]
    feats["diff_ko_taken"] = feats["p1_n_ko_taken"] - feats["p2_n_ko_taken"]
    return feats


# ---------------------------------------------------------------------------
# Final model: what the players did, and who is still standing
# ---------------------------------------------------------------------------

def move_usage_features(battle, context):
    """How many times each side used each move in the 30 turns.

    The turn-30 state says who is ahead. The moves say how each side plays:
    stalling with Rest and Toxic, setting Spikes, going for Explosion trades.
    The list of moves comes from the training battles (context["move_vocab"]).
    """
    feats = {}
    for side in ("p1", "p2"):
        used = Counter(t[f"{side}_move_details"]["name"]
                       for t in battle["battle_timeline"] if t[f"{side}_move_details"])
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
            if move and move["category"] != "status" and before["name"] == after["name"]:
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

        damage = np.array(damage) if damage else np.array([0.0])
        feats[f"{me}_dmg_mean"] = damage.mean()
        feats[f"{me}_dmg_total"] = damage.clip(min=0).sum()
        feats[f"{me}_dmg_max"] = damage.max()
        feats[f"{me}_n_hits"] = len(damage)
        feats[f"{me}_no_effect_hits"] = no_effect
        feats[f"{me}_heal_total"] = healed
        feats[f"{me}_turns_lost"] = lost
        feats[f"{me}_turns_slp"] = asleep
        feats[f"{me}_turns_par"] = paralysed

    for key in ("dmg_mean", "dmg_total", "dmg_max", "heal_total", "turns_lost",
                "turns_slp", "turns_par", "no_effect_hits"):
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


# ---------------------------------------------------------------------------
# Compact model: three Gen 2 roles instead of the full move list
# ---------------------------------------------------------------------------

SPIKES = {"spikes"}
SPIN = {"rapidspin"}
BOOM = {"explosion", "selfdestruct"}


def roles_features(battle):
    """How many Pokemon on each side showed three key Gen 2 roles.

    Spikes stays on the field until Rapid Spin removes it, and Explosion
    trades one Pokemon for another. Counting the Pokemon that used these
    moves is a short summary of what the full move list says about a team.
    """
    feats = {}
    for side in ("p1", "p2"):
        used = defaultdict(set)
        for turn in battle["battle_timeline"]:
            move = turn[f"{side}_move_details"]
            if move:
                used[turn[f"{side}_pokemon_state"]["name"]].add(move["name"])
        states = last_seen_states(battle, side)
        feats[f"{side}_spikes_users"] = sum(1 for m in used.values() if m & SPIKES)
        feats[f"{side}_spin_users"] = sum(1 for m in used.values() if m & SPIN)
        feats[f"{side}_boom_users"] = sum(1 for m in used.values() if m & BOOM)
        feats[f"{side}_spin_alive"] = sum(1 for n, m in used.items()
                                          if m & SPIN and states[n]["status"] != "fnt")
    feats["diff_spin_alive"] = feats["p1_spin_alive"] - feats["p2_spin_alive"]
    return feats


# ---------------------------------------------------------------------------
# Team records: which known team each side is playing
# ---------------------------------------------------------------------------

def team_id_features(battle, context):
    """Ids of the two teams, so the model can look up their past results.

    P1's team is known in full. For P2 we take the known teams that contain
    every P2 Pokemon seen in the 30 turns and keep the most common one. The
    ids themselves are not used as features: TeamRecordModel (models.py)
    turns them into win records inside each training fold.
    """
    index = context["teams"]
    key = tuple(sorted(p["name"] for p in battle["p1_team_details"]))
    seen = {turn["p2_pokemon_state"]["name"] for turn in battle["battle_timeline"]}
    candidates = set.intersection(*(index["by_species"].get(n, set()) for n in seen))
    p2 = max(candidates, key=lambda i: index["count"][i]) if candidates else -1
    return {"p1_team_id": index["ids"].get(key, -1), "p2_team_id": p2,
            "p2_team_candidates": len(candidates)}


# ---------------------------------------------------------------------------
# Registry: name used in experiment configs -> function
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "starter": starter_features,
    "state_t30": state_t30_features,
    "trajectory": trajectory_features,
    "matchup": matchup_features,
    "sacrifice": sacrifice_features,
    "move_usage": move_usage_features,
    "move_effects": move_effect_features,
    "alive_species": alive_species_features,
    "roles": roles_features,
    "team_ids": team_id_features,
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
