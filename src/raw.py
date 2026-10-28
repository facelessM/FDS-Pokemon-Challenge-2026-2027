"""A separate view of the recorded rows, without inferring missing actions."""

from collections import Counter, defaultdict
import numpy as np
import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]


BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


STATUSES = ["nostatus", "slp", "par", "psn", "tox", "brn", "frz", "fnt"]


def token(value):
    return str(value).replace(" ", "_") if value is not None else "missing"


def snapshot(row, side):
    state = row[side + "_pokemon_state"]
    move = row[side + "_move_details"]
    effects = [str(e) for e in state.get("effects", []) if e != "noeffect"]
    result = {
        "c_species": token(state["name"]),
        "c_move": token(move["name"]) if move else "missing",
        "c_move_type": token(move["type"]) if move else "missing",
        "c_effects": "|".join(sorted(effects)) or "none",
        "hp": float(state["hp_pct"]),
        "move_present": float(move is not None),
        "move_power": float(move["base_power"]) if move else 0.0,
        "move_accuracy": float(move["accuracy"]) if move else 0.0,
        "move_priority": float(move["priority"]) if move else 0.0,
        "effects_count": float(len(effects)),
    }
    for status in STATUSES:
        result["status_" + status] = float(state.get("status", "nostatus") == status)
    for name in BOOSTS:
        result["boost_" + name] = float(state.get("boosts", {}).get(name, 0))
    for name in ["status", "physical", "special"]:
        result["move_category_" + name] = float(bool(move) and move["category"] == name)
    return result


def attach(out, prefix, values):
    for key, value in values.items():
        out[("c_" + prefix + key[2:]) if key.startswith("c_") else prefix + key] = value


def summary(battle):
    rows = battle["battle_timeline"]
    out = {}
    out["c_p1_lead"] = token(battle["p1_team_details"][0]["name"])
    for i, pokemon in enumerate(
        sorted(battle["p1_team_details"], key=lambda p: p["name"])
    ):
        pre = f"p1_roster_{i}_"
        out["c_" + pre + "species"] = token(pokemon["name"])
        out["c_" + pre + "item"] = token(pokemon.get("item", "missing"))
        out["c_" + pre + "types"] = "|".join(sorted(pokemon["types"]))
        out[pre + "level"] = float(pokemon["level"])
        for stat in STATS:
            out[pre + stat] = float(pokemon["base_" + stat])
    lead = battle["p2_lead_details"]
    out["c_p2_lead"] = token(lead["name"])
    out["c_p2_lead_item"] = token(lead.get("item", "missing"))
    out["c_p2_lead_types"] = "|".join(sorted(lead["types"]))
    out["p2_lead_level"] = float(lead["level"])
    for stat in STATS:
        out["p2_lead_" + stat] = float(lead["base_" + stat])
    for side in ["p1", "p2"]:
        snapshots = [snapshot(row, side) for row in rows]
        numeric = [key for key in snapshots[0] if not key.startswith("c_")]
        for key in numeric:
            values = np.array([s[key] for s in snapshots])
            for label, value in [
                ("mean", values.mean()),
                ("min", values.min()),
                ("max", values.max()),
            ]:
                out[f"{side}_pool_{key}_{label}"] = float(value)
        attach(out, side + "_active_", snapshots[-1])
        states = {}
        last = {}
        moves = defaultdict(set)
        counts = Counter()
        all_moves = Counter()
        for i, row in enumerate(rows):
            state = row[side + "_pokemon_state"]
            move = row[side + "_move_details"]
            name = state["name"]
            states[name] = state
            last[name] = i
            counts[name] += 1
            if move:
                moves[name].add(move["name"])
                all_moves[move["name"]] += 1
        names = sorted(states)
        out[side + "_observed_species"] = float(len(names))
        out[side + "_distinct_moves"] = float(len(all_moves))
        out[side + "_distinct_species_moves"] = float(
            sum(len(moves[name]) for name in names)
        )
        out[side + "_observed_hp_sum"] = float(
            sum(s["hp_pct"] for s in states.values())
        )
        out[side + "_explicit_faints"] = float(
            sum(s["status"] == "fnt" for s in states.values())
        )
        for slot in range(6):
            pre = f"{side}_seen_{slot}_"
            present = slot < len(names)
            name = names[slot] if present else None
            state = states.get(name, {})
            out[pre + "present"] = float(present)
            out["c_" + pre + "species"] = token(name)
            out["c_" + pre + "moves"] = (
                "|".join(sorted(moves.get(name, set()))) or "missing"
            )
            out[pre + "hp"] = float(state.get("hp_pct", 0))
            out[pre + "age"] = float(29 - last[name]) if present else 30.0
            out[pre + "observed_rows"] = float(counts.get(name, 0))
            out[pre + "move_count"] = float(len(moves.get(name, set())))
            for status in STATUSES:
                out[pre + "status_" + status] = float(
                    present and state.get("status") == status
                )
    for key in list(out):
        if key.startswith("p1_") and "p2_" + key[3:] in out:
            out["diff_" + key[3:]] = out[key] - out["p2_" + key[3:]]
    return out


def frame(rows):
    result = pd.DataFrame(rows)
    for c in result:
        result[c] = (
            result[c].fillna("missing").astype(str)
            if c.startswith("c_")
            else result[c].fillna(0).astype(np.float32)
        )
    assert result.columns.is_unique
    numeric = result.select_dtypes(include=np.number)
    assert np.isfinite(numeric.to_numpy()).all()
    return result
