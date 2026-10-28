"""Learn departure evidence on P1, where attack knockouts are recorded.

The scores transferred to P2 are uncertain features, not confirmed deaths.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from .state import pressure, type_mult, BOOM, STATUS


def make_events(B, V, stats, chart, include_entry=False):
    rows = []
    meta = []
    for bi, (b, v) in enumerate(zip(B, V)):
        tl = b["battle_timeline"]
        for side, opp in [("p1", "p2"), ("p2", "p1")]:
            for i in range(1, len(tl)):
                before = tl[i - 1][side + "_pokemon_state"]
                now = tl[i][side + "_pokemon_state"]
                n = before["name"]
                if before["status"] == "fnt" or before["hp_pct"] <= 0:
                    continue
                positive = side == "p1" and now["name"] == n and now["status"] == "fnt"
                gone = now["name"] != n and v["last_at"][side][n] == i - 1
                if not positive and not gone:
                    continue
                if positive:
                    ownmove = tl[i][side + "_move_details"]
                    if ownmove and ownmove["name"] in BOOM:
                        continue
                entry_index = i
                if include_entry and positive:
                    entry_index = next(
                        (
                            j
                            for j in range(i + 1, len(tl))
                            if tl[j][side + "_pokemon_state"]["name"] != n
                        ),
                        None,
                    )
                    if entry_index is None:
                        continue
                enemy0 = tl[i - 1][opp + "_pokemon_state"]
                enemy1 = tl[i][opp + "_pokemon_state"]
                m = tl[i][opp + "_move_details"]
                oldm = tl[i - 1][opp + "_move_details"]
                mine = tl[i - 1][side + "_move_details"]
                a = stats.get(enemy1["name"], {})
                d = stats.get(n, {})
                f = {
                    "victim": n,
                    "attacker": enemy1["name"],
                    "attack_move": m["name"] if m else "_none",
                    "previous_own_move": mine["name"] if mine else "_none",
                    "hp": before["hp_pct"],
                    "age": 29 - i,
                    "turn": i,
                    "enemy_hp": enemy0["hp_pct"],
                    "enemy_hp_after": enemy1["hp_pct"],
                    "enemy_same": int(enemy0["name"] == enemy1["name"]),
                    "enemy_hp_change": (
                        enemy1["hp_pct"] - enemy0["hp_pct"]
                        if enemy0["name"] == enemy1["name"]
                        else 0.0
                    ),
                    "attack_changed": int(m != oldm),
                    "attack_power": m["base_power"] if m else 0.0,
                    "attack_accuracy": m["accuracy"] if m else 0.0,
                    "attack_priority": m["priority"] if m else 0.0,
                    "attack_category": (
                        {"status": 1, "physical": 2, "special": 3}.get(m["category"], 0)
                        if m
                        else 0
                    ),
                    "own_prior_power": mine["base_power"] if mine else 0.0,
                    "unknown_power": int(
                        bool(m and m["category"] != "status" and m["base_power"] == 0)
                    ),
                    "type_effectiveness": (
                        type_mult(m["type"], d, chart) if m and d else 1.0
                    ),
                    "pressure": (
                        pressure(m, a, d, enemy1, before, chart) if a and d else 0.0
                    ),
                }
                f["pressure_hp_ratio"] = min(
                    8.0, f["pressure"] / max(before["hp_pct"], 0.03)
                )
                for label, state in [("own", before), ("enemy", enemy1)]:
                    for status in STATUS[2:]:
                        f[label + "_status_" + status] = int(state["status"] == status)
                    for stat in ["atk", "def", "spa", "spd", "spe"]:
                        f[label + "_boost_" + stat] = state["boosts"].get(stat, 0)
                    for effect in [
                        "substitute",
                        "perish",
                        "trapped",
                        "leechseed",
                        "confusion",
                        "encore",
                    ]:
                        f[label + "_effect_" + effect] = int(effect in state["effects"])
                for label, poke in [("own", d), ("enemy", a)]:
                    for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
                        f[label + "_base_" + stat] = poke.get("base_" + stat, 0)
                if i >= 2 and tl[i - 2][side + "_pokemon_state"]["name"] == n:
                    f["previous_hp_loss"] = (
                        tl[i - 2][side + "_pokemon_state"]["hp_pct"] - before["hp_pct"]
                    )
                else:
                    f["previous_hp_loss"] = 0.0
                if include_entry:
                    incoming = tl[entry_index][side + "_pokemon_state"]
                    incname = incoming["name"]
                    previous = [
                        t[side + "_pokemon_state"]
                        for t in tl[:entry_index]
                        if t[side + "_pokemon_state"]["name"] == incname
                    ]
                    last_hp = previous[-1]["hp_pct"] if previous else 1.0
                    loss = last_hp - incoming["hp_pct"]
                    incstats = stats.get(incname, {})
                    f.update(
                        incoming_hp=incoming["hp_pct"],
                        incoming_prior_hp=last_hp,
                        incoming_loss=loss,
                        incoming_new=int(not previous),
                        incoming_full=int(incoming["hp_pct"] >= 0.99),
                        incoming_grounded=int(
                            "flying" not in incstats.get("types", [])
                        ),
                        incoming_eighth=int(abs(loss - 0.125) < 0.025),
                        incoming_sixteenth=int(abs(loss - 0.0625) < 0.02),
                        incoming_immune=int(
                            bool(
                                m
                                and incstats
                                and type_mult(m["type"], incstats, chart) == 0
                            )
                        ),
                        incoming_pressure=(
                            pressure(m, a, incstats, enemy1, None, chart)
                            if a and incstats
                            else 0.0
                        ),
                    )
                    f["incoming_pressure_gap"] = f["incoming_pressure"] - max(loss, 0)
                rows.append(f)
                meta.append(
                    (
                        bi,
                        side,
                        n,
                        before["hp_pct"],
                        29 - i,
                        int(positive) if side == "p1" else -1,
                    )
                )
    X = pd.DataFrame(rows).fillna(0)
    M = pd.DataFrame(meta, columns=["battle", "side", "species", "hp", "age", "target"])
    return X, M


def model():
    return lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=3,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        n_jobs=2,
        random_state=0,
        verbosity=-1,
    )


def aggregate(V, M, p, top):
    rows = []
    by = defaultdict(list)
    for index, row in M[M.side == "p2"].iterrows():
        by[int(row.battle)].append(
            (row.species, float(p[index]), float(row.hp), float(row.age))
        )
    for bi, v in enumerate(V):
        candidates = by[bi]
        risk = [z[1] for z in candidates]
        st = v["states"]["p2"]
        alive = {n: s for n, s in st.items() if s["status"] != "fnt"}
        oldhp = sum(s["hp_pct"] for s in alive.values()) + max(6 - len(st), 0)
        f = {
            "ko_candidates": len(candidates),
            "ko_soft_count": sum(risk),
            "ko_risk_max": max(risk, default=0),
            "ko_any": 1 - np.prod([1 - q for q in risk]),
            "ko_lost_hp": sum(q * h for n, q, h, age in candidates),
            "ko_risk_age": sum(q * age / 30 for n, q, h, age in candidates),
            "ko_above_half": sum(q >= 0.5 for q in risk),
            "ko_above_eight": sum(q >= 0.8 for q in risk),
            "ko_corrected_alive": 6 - (len(st) - len(alive)) - sum(risk),
            "ko_recoverer_risk": sum(
                q
                for n, q, h, age in candidates
                if "rest" in v["moves"]["p2"].get(n, [])
            ),
        }
        f["ko_corrected_hp"] = oldhp - f["ko_lost_hp"]
        for j, q in enumerate(sorted(risk, reverse=True)[:3] + [0.0] * 3):
            if j < 3:
                f["ko_risk_rank" + str(j)] = q
        lookup = {n: (q, h) for n, q, h, age in candidates}
        for n in top:
            q, h = lookup.get(n, (0.0, alive.get(n, {}).get("hp_pct", 0.0)))
            f["ko_p2_risk_" + n] = q
            f["ko_p2_adjusted_hp_" + n] = (1 - q) * h
        rows.append(f)
    return pd.DataFrame(rows).astype(np.float32)
