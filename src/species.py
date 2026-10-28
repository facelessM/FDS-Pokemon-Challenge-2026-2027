"""Attach HP to its species and build observations at five visible prefixes."""

from collections import Counter
import numpy as np
import pandas as pd

STATUS = ["nostatus", "fnt", "slp", "par", "tox", "psn", "frz", "brn"]


RECOVERY = {
    "rest",
    "recover",
    "softboiled",
    "milkdrink",
    "synthesis",
    "moonlight",
    "morningsun",
}


ROLES = {
    "rest": {"rest"},
    "talk": {"sleeptalk"},
    "recovery": RECOVERY,
    "boom": {"explosion", "selfdestruct"},
    "spikes": {"spikes"},
    "spin": {"rapidspin"},
    "phaze": {"roar", "whirlwind"},
    "bell": {"healbell"},
}


def vocabulary(views):
    species = Counter(n for v in views for n in v["team"])
    top = [n for n, _ in sorted(species.items(), key=lambda z: (-z[1], z[0]))[:20]]
    pairs = Counter()
    for v in views:
        pairs.update(
            {
                (n, m)
                for side in ["p1", "p2"]
                for n, ms in v["moves"][side].items()
                for m in ms
            }
        )
    tokens = [
        x for x, c in sorted(pairs.items(), key=lambda z: (-z[1], z[0])) if c >= 100
    ][:96]
    return top, tokens


def hp(v, side, name):
    st = v["states"][side].get(name)
    if st:
        return 0.0 if st["status"] == "fnt" else st["hp_pct"]
    return float(side == "p1" and name in v["team"])


def species_features(v, top):
    out = {}
    for n in top:
        for side in ["p1", "p2"]:
            st = v["states"][side].get(n)
            h = hp(v, side, n)
            state = st["status"] if st else "nostatus"
            stem = f"owned_{side}_{n}_"
            out[stem + "hp"] = h
            out[stem + "sleep_hp"] = h * (state == "slp")
            out[stem + "paralysis_hp"] = h * (state == "par")
            out[stem + "other_status_hp"] = h * (state in ["tox", "psn", "brn", "frz"])
            out[stem + "observed"] = int(st is not None)
            out[stem + "age"] = (29 - v["last_at"][side][n]) / 30 if st else 1.0
        for key in ["hp", "sleep_hp", "paralysis_hp", "other_status_hp"]:
            out[f"owned_diff_{n}_{key}"] = (
                out[f"owned_p1_{n}_{key}"] - out[f"owned_p2_{n}_{key}"]
            )
    return out


def snapshots(b):
    states = {"p1": {}, "p2": {}}
    moves = {"p1": {}, "p2": {}}
    last_at = {"p1": {}, "p2": {}}
    totals = {s: [] for s in states}
    result, targets = [], []
    tl = b["battle_timeline"]
    for i, turn in enumerate(tl):
        for side in states:
            st, m = turn[side + "_pokemon_state"], turn[side + "_move_details"]
            states[side][st["name"]] = st
            last_at[side][st["name"]] = i
            if m:
                moves[side].setdefault(st["name"], set()).add(m["name"])
            total = sum(
                s["hp_pct"] for s in states[side].values() if s["status"] != "fnt"
            ) + max(0, 6 - len(states[side]))
            totals[side].append(total)
        if i + 1 not in [10, 15, 20, 25, 30]:
            continue
        row = {}
        for side in states:
            pre = side + "_"
            alive = [s for s in states[side].values() if s["status"] != "fnt"]
            active = turn[side + "_pokemon_state"]
            name = active["name"]
            ms = moves[side].get(name, set())
            move = turn[side + "_move_details"]
            row[pre + "species"] = name
            row[pre + "last_move"] = move["name"] if move else ""
            row[pre + "total_hp"] = totals[side][-1]
            row[pre + "hp_sq"] = sum(s["hp_pct"] ** 2 for s in alive) + max(
                0, 6 - len(states[side])
            )
            row[pre + "seen"] = len(states[side])
            row[pre + "fnt"] = sum(s["status"] == "fnt" for s in states[side].values())
            row[pre + "low_hp"] = sum(s["hp_pct"] < 0.25 for s in alive)
            row[pre + "active_hp"] = (
                active["hp_pct"] if active["status"] != "fnt" else 0.0
            )
            row[pre + "revealed_moves"] = sum(len(z) for z in moves[side].values())
            row[pre + "stale_hp"] = sum(
                s["hp_pct"]
                for n, s in states[side].items()
                if s["status"] != "fnt" and i - last_at[side][n] >= 5
            )
            for status in STATUS:
                row[pre + status + "_count"] = sum(
                    s["status"] == status for s in states[side].values()
                )
                row[pre + "active_" + status] = int(active["status"] == status)
            for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
                row[pre + "boost_" + stat] = active["boosts"].get(stat, 0)
            for effect in [
                "substitute",
                "trapped",
                "perish",
                "leechseed",
                "confusion",
                "encore",
            ]:
                row[pre + effect] = int(effect in active["effects"])
            for role, names in ROLES.items():
                row[pre + "active_" + role] = int(bool(ms & names))
                row[pre + role + "_hp"] = sum(
                    s["hp_pct"]
                    for n, s in states[side].items()
                    if s["status"] != "fnt" and moves[side].get(n, set()) & names
                )
            for width in [3, 5]:
                row[pre + f"hp_delta{width}"] = (
                    totals[side][i] - totals[side][i - width]
                )
                row[pre + f"switches{width}"] = sum(
                    tl[j][side + "_pokemon_state"]["name"]
                    != tl[j - 1][side + "_pokemon_state"]["name"]
                    for j in range(i - width + 1, i + 1)
                )
        for key in list(row):
            if key.startswith("p1_") and not isinstance(row[key], str):
                row["diff_" + key[3:]] = row[key] - row[key.replace("p1_", "p2_", 1)]
        result.append(row)
        targets.append([totals[s][-1] for s in ["p1", "p2"]])
    level = np.asarray(targets)
    return result, level[1:] - level[:-1], level


def matrix(rows):
    frame = pd.DataFrame(rows).fillna(0).astype(np.float32)
    assert frame.columns.is_unique and np.isfinite(frame.to_numpy()).all()
    return frame
