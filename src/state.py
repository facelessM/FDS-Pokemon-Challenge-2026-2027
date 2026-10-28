"""Recovery resources and departure evidence from the visible thirty turns."""

from collections import defaultdict
import math
import numpy as np

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


BOOM = {"explosion", "selfdestruct"}


PHASE = {"roar", "whirlwind"}


def stats_context(battles):
    return {
        p["name"]: p
        for b in battles
        for p in b["p1_team_details"] + [b["p2_lead_details"]]
    }


def stages(s):
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


def type_mult(move, pokemon, chart):
    return math.prod(chart.get((move, t), 1.0) for t in pokemon.get("types", []))


def pressure(move, a, d, sa=None, sd=None, chart=None):
    """Relative pressure, not an exact damage calculation."""
    if not move or move["category"] == "status" or not a or not d:
        return 0.0
    at, de = ("atk", "def") if move["category"] == "physical" else ("spa", "spd")
    value = (
        move["base_power"]
        * type_mult(move["type"], d, chart)
        * (1.5 if move["type"] in a["types"] else 1.0)
    )
    value *= (
        (2 * a["base_" + at] + 100)
        / (2 * d["base_" + de] + 100)
        / (2 * d["base_hp"] + 200)
    )
    if sa:
        value *= stages(sa.get("boosts", {}).get(at, 0)) * (
            0.5 if sa["status"] == "brn" and at == "atk" else 1.0
        )
    if sd:
        value /= stages(sd.get("boosts", {}).get(de, 0))
    return value


def aggregate(out, prefix, values):
    a = np.array(values, float)
    out[prefix + "_n"] = len(a)
    out[prefix + "_sum"] = float(a.sum())
    out[prefix + "_mean"] = float(a.mean()) if len(a) else 0.0
    out[prefix + "_max"] = float(a.max()) if len(a) else 0.0


def differences(out, group):
    stem = "p1_" + group + "_"
    for k in list(out):
        if k.startswith(stem):
            out["diff_" + group + "_" + k[len(stem) :]] = out[k] - out.get(
                "p2_" + group + "_" + k[len(stem) :], 0.0
            )
    return out


def departures(b, v, stats, chart):
    out = {}
    tl = b["battle_timeline"]
    for side, opp in [("p1", "p2"), ("p2", "p1")]:
        prefix = side + "_exit_"
        candidates = []
        vals = defaultdict(list)
        for i in range(1, len(tl)):
            prev = tl[i - 1][side + "_pokemon_state"]
            cur = tl[i][side + "_pokemon_state"]
            name = prev["name"]
            if (
                name == cur["name"]
                or prev["status"] == "fnt"
                or v["last_at"][side][name] != i - 1
            ):
                continue
            before = tl[i - 1][opp + "_pokemon_state"]
            after = tl[i][opp + "_pokemon_state"]
            m = tl[i][opp + "_move_details"]
            age = 30 - i
            new = cur["name"] not in {
                t[side + "_pokemon_state"]["name"] for t in tl[:i]
            }
            same = before["name"] == after["name"]
            hit = bool(m and m["category"] != "status")
            oldhp = prev["hp_pct"]
            restore = bool(v["moves"][side].get(name, set()) & RECOVERY)
            pp = (
                pressure(
                    m, stats.get(after["name"]), stats.get(name), after, prev, chart
                )
                if same
                else 0.0
            )
            candidates.append(oldhp)
            for threshold in [0.15, 0.3, 0.5]:
                key = f"low{threshold}"
                vals[key].append(float(oldhp <= threshold))
                vals[key + "_fresh_incoming"].append(
                    float(oldhp <= threshold and new and cur["hp_pct"] >= 0.98)
                )
            vals["age"].append(age)
            vals["hp_age"].append(oldhp * age / 30)
            vals["low_no_recovery"].append(float(oldhp < 0.4 and not restore))
            vals["pressure_to_hp"].append(min(8.0, pp / max(oldhp, 0.05)))
            vals["full_incoming_attack"].append(
                float(new and cur["hp_pct"] >= 0.98 and same and hit)
            )
            vals["attack_no_incoming_damage"].append(
                float(cur["hp_pct"] >= 0.98 and same and hit)
            )
            vals["opp_phaze"].append(float(m is not None and m["name"] in PHASE))
            vals["recoverable_hp_gap"].append((1 - oldhp) * restore)
            vals["opp_hp_loss"].append(
                max(0.0, before["hp_pct"] - after["hp_pct"]) if same else 0.0
            )
        aggregate(out, prefix + "hp", candidates)
        for key in [
            "age",
            "hp_age",
            "low0.15",
            "low0.15_fresh_incoming",
            "low0.3",
            "low0.3_fresh_incoming",
            "low0.5",
            "low0.5_fresh_incoming",
            "low_no_recovery",
            "pressure_to_hp",
            "full_incoming_attack",
            "attack_no_incoming_damage",
            "opp_phaze",
            "recoverable_hp_gap",
            "opp_hp_loss",
        ]:
            values = vals[key]
            out[prefix + key + "_sum"] = sum(values)
            out[prefix + key + "_max"] = max(values, default=0.0)
    return differences(out, "exit")


def resources(b, v):
    out = {}
    tl = b["battle_timeline"]
    for side in ["p1", "p2"]:
        pre = side + "_resource_"
        states = v["states"][side]
        alive = {n: s for n, s in states.items() if s["status"] != "fnt"}
        unseen = max(0, 6 - len(states))
        hp = sorted(
            [s["hp_pct"] for s in alive.values()] + [1.0] * unseen, reverse=True
        )
        hp = hp + [0.0] * (6 - len(hp))
        for j, h in enumerate(hp[:6]):
            out[pre + f"hp_rank{j}"] = h
        out[pre + "hp_entropy"] = -sum(
            (h / max(sum(hp), 1e-8)) * math.log(h / max(sum(hp), 1e-8))
            for h in hp
            if h > 0
        )
        out[pre + "healthy_count"] = sum(h > 0.75 for h in hp)
        out[pre + "near_dead_count"] = sum(0 < h < 0.15 for h in hp)
        vals = defaultdict(list)
        for n, s in alive.items():
            ms = v["moves"][side].get(n, set())
            h = s["hp_pct"]
            age = 29 - v["last_at"][side][n]
            rest = "rest" in ms
            talk = "sleeptalk" in ms
            other_recovery = bool(ms & (RECOVERY - {"rest"}))
            recover = rest or other_recovery
            vals["hp_with_recovery"].append(h * recover)
            vals["hp_without_recovery"].append(h * (not recover))
            vals["recoverable_deficit"].append((1 - h) * recover)
            vals["resttalk_deficit"].append((1 - h) * rest * talk)
            vals["low_no_recovery"].append(float(h < 0.4 and not recover))
            vals["low_recovery"].append(float(h < 0.4 and recover))
            vals["sleep_resttalk"].append(h * (s["status"] == "slp") * rest * talk)
            vals["sleep_no_talk"].append(h * (s["status"] == "slp") * (not talk))
            vals["toxic_no_rest"].append(h * (s["status"] == "tox") * (not rest))
            vals["paralyzed_hp"].append(h * (s["status"] == "par"))
            vals["fresh_status_hp"].append(
                h * (s["status"] not in ["nostatus", "fnt"]) * (age <= 3)
            )
            for role, move in [
                ("spin", "rapidspin"),
                ("spikes", "spikes"),
                ("bell", "healbell"),
                ("boom", None),
                ("setup", "curse"),
            ]:
                has = bool(ms & BOOM) if move is None else move in ms
                vals[role + "_healthy"].append(float(has and h > 0.5))
                vals[role + "_critical"].append(float(has and h < 0.25))
        for key in [
            "hp_with_recovery",
            "hp_without_recovery",
            "recoverable_deficit",
            "resttalk_deficit",
            "low_no_recovery",
            "low_recovery",
            "sleep_resttalk",
            "sleep_no_talk",
            "toxic_no_rest",
            "paralyzed_hp",
            "fresh_status_hp",
        ] + [
            r + "_" + q
            for r in ["spin", "spikes", "bell", "boom", "setup"]
            for q in ["healthy", "critical"]
        ]:
            out[pre + key] = sum(vals[key])
        active = tl[-1][side + "_pokemon_state"]
        n = active["name"]
        streak = 0
        rest_start = 0
        hp_jump = 0.0
        for j in range(len(tl) - 1, -1, -1):
            q = tl[j][side + "_pokemon_state"]
            if q["name"] != n or q["status"] != active["status"]:
                break
            streak += 1
            if j:
                old = tl[j - 1][side + "_pokemon_state"]
                m = tl[j][side + "_move_details"]
                if old["name"] == n and old["status"] != "slp" and q["status"] == "slp":
                    rest_start = int(bool(m and m["name"] == "rest"))
                    hp_jump = q["hp_pct"] - old["hp_pct"]
        out[pre + "active_sleep_streak"] = streak if active["status"] == "slp" else 0
        out[pre + "active_sleep_rest_onset"] = rest_start
        out[pre + "active_sleep_hp_jump"] = hp_jump
        out[pre + "active_toxic_streak"] = streak if active["status"] == "tox" else 0
        for effect in [
            "perish",
            "encore",
            "substitute",
            "trapped",
            "leechseed",
            "confusion",
        ]:
            clock = 0
            for q in reversed(tl):
                z = q[side + "_pokemon_state"]
                if z["name"] != n or effect not in z["effects"]:
                    break
                clock += 1
            out[pre + "active_" + effect + "_clock"] = clock
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            out[pre + "hp_boost_" + stat] = active["hp_pct"] * active["boosts"].get(
                stat, 0
            )
    return differences(out, "resource")


def switches(b, v, stats, chart):
    out = {}
    tl = b["battle_timeline"]
    for side, opp in [("p1", "p2"), ("p2", "p1")]:
        pre = side + "_switch_"
        vals = defaultdict(list)
        last = {}
        hazard = 0.0
        hazevidence = []
        ghost = False
        for i, t in enumerate(tl):
            own = t[side + "_pokemon_state"]
            enemy = t[opp + "_pokemon_state"]
            om = t[opp + "_move_details"]
            m = t[side + "_move_details"]
            if om and om["name"] == "spikes":
                hazard = 1.0
            if (
                m
                and m["name"] == "rapidspin"
                and "ghost" not in stats.get(enemy["name"], {}).get("types", [])
                and i
                and own["name"] == tl[i - 1][side + "_pokemon_state"]["name"]
            ):
                hazard = 0.25
            if i:
                before = tl[i - 1][side + "_pokemon_state"]
                old_enemy = tl[i - 1][opp + "_pokemon_state"]
                oldmove = tl[i - 1][opp + "_move_details"]
                if own["name"] != before["name"]:
                    same = old_enemy["name"] == enemy["name"]
                    a = stats.get(old_enemy["name"], {})
                    old = stats.get(before["name"], {})
                    new = stats.get(own["name"], {})
                    if same and old and new and oldmove:
                        po = pressure(oldmove, a, old, old_enemy, before, chart)
                        pn = pressure(oldmove, a, new, old_enemy, None, chart)
                        vals["pressure_saved"].append(po - pn)
                        vals["low_hp_saved"].append((po - pn) * (1 - before["hp_pct"]))
                        vals["resist_entry"].append(
                            float(
                                type_mult(oldmove["type"], new, chart) < 1
                                and oldmove["category"] != "status"
                            )
                        )
                        vals["immune_entry"].append(
                            float(
                                type_mult(oldmove["type"], new, chart) == 0
                                and oldmove["category"] != "status"
                            )
                        )
                        vals["worse_entry"].append(float(pn > po + 0.05))
                    prior = last.get(own["name"], {"hp_pct": 1.0})["hp_pct"]
                    loss = prior - own["hp_pct"]
                    grounded = "flying" not in new.get("types", [])
                    vals["entry_hp_loss"].append(max(0, loss))
                    vals["ground_entry_loss"].append(max(0, loss) * grounded)
                    vals["flying_entry_loss"].append(max(0, loss) * (not grounded))
                    vals["full_new_entry"].append(
                        float(own["name"] not in last and own["hp_pct"] == 1)
                    )
                    if grounded and (
                        abs(loss - 0.125) < 0.025 or abs(loss - 0.0625) < 0.02
                    ):
                        hazevidence.append((i, 1.0))
                    if not grounded and abs(loss) < 0.02:
                        vals["flying_clean"].append(1.0)
                    if same and om and om["name"] in PHASE:
                        vals["phazed"].append(1.0)
                    if not same:
                        vals["double_switch"].append(1.0)
            last[own["name"]] = own
        for key in [
            "pressure_saved",
            "low_hp_saved",
            "resist_entry",
            "immune_entry",
            "worse_entry",
            "entry_hp_loss",
            "ground_entry_loss",
            "flying_entry_loss",
            "full_new_entry",
            "flying_clean",
            "phazed",
            "double_switch",
        ]:
            aggregate(out, pre + key, vals[key])
        out[pre + "hazard_token_state"] = hazard
        out[pre + "hazard_entry_evidence"] = len(hazevidence)
        out[pre + "hazard_recent_evidence"] = sum(i >= 20 for i, z in hazevidence)
        alive = {n: s for n, s in v["states"][side].items() if s["status"] != "fnt"}
        out[pre + "grounded_critical"] = sum(
            "flying" not in stats.get(n, {}).get("types", []) and s["hp_pct"] < 0.13
            for n, s in alive.items()
        )
        out[pre + "hazard_critical_pressure"] = hazard * out[pre + "grounded_critical"]
    return differences(out, "switch")
