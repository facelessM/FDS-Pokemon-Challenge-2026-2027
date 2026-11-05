"""Move constraints, active matchups, and remaining defensive coverage.

The response scores are uncertain summaries, not a battle simulator.
All species, move metadata, set examples and HP-response estimates come from train.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
from .tactics import TacticalContext, stage
from .options import FEATURE_FIXES

BOOM = {"explosion", "selfdestruct"}
COPY = {"mimic", "metronome", "transform", "sketch"}
HEAL = {
    "rest",
    "recover",
    "softboiled",
    "milkdrink",
    "moonlight",
    "morningsun",
    "synthesis",
}
PHAZE = {"roar", "whirlwind"}
RECOVERY_BLOCK = HEAL | {
    "sleeptalk",
    "protect",
    "substitute",
    "gigadrain",
    "megadrain",
    "dreameater",
    "absorb",
}
RESIDUAL = {
    "substitute",
    "leechseed",
    "curse",
    "nightmare",
    "firespin",
    "whirlpool",
    "clamp",
    "wrap",
    "bind",
}
STATUS = ["nostatus", "slp", "par", "psn", "tox", "frz", "brn", "fnt"]
MODES = ["consistent"]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def summaries(values):
    x = np.asarray(values, float)
    return {
        "sum": float(x.sum()),
        "max": float(x.max()) if x.size else 0.0,
        "mean": float(x.mean()) if x.size else 0.0,
    }


def differences(f, stem):
    for key in list(f):
        if key.startswith("p1_" + stem):
            f["diff_" + key[3:]] = f[key] - f.get("p2_" + key[3:], 0.0)
    return f


def evidence(battles, views, ctx):
    """Conservative continuous intervals; their target is net HP loss."""
    rows = []
    ratios = []
    for battle, view in zip(battles, views):
        for old, cur in zip(battle["battle_timeline"], battle["battle_timeline"][1:]):
            if any(
                old[s + "_pokemon_state"]["name"] != cur[s + "_pokemon_state"]["name"]
                for s in ["p1", "p2"]
            ):
                continue
            for side, other in [("p1", "p2"), ("p2", "p1")]:
                a, aa = old[side + "_pokemon_state"], cur[side + "_pokemon_state"]
                b, bb = old[other + "_pokemon_state"], cur[other + "_pokemon_state"]
                move, reply = cur[side + "_move_details"], cur[other + "_move_details"]
                if not move or move["category"] == "status" or move["name"] in BOOM:
                    continue
                if move == old[side + "_move_details"]:
                    continue
                if a["status"] in {"slp", "frz", "fnt"} or b["status"] not in {
                    "nostatus",
                    "par",
                }:
                    continue
                if (
                    b["hp_pct"] < 0.25
                    or bb["hp_pct"] <= 0
                    or bb["hp_pct"] > b["hp_pct"]
                ):
                    continue
                if reply and reply["name"] in RECOVERY_BLOCK:
                    continue
                if any(RESIDUAL & set(st["effects"]) for st in [a, aa, b, bb]):
                    continue
                if any(any(st["boosts"].values()) for st in [a, aa, b, bb]):
                    continue
                loss = b["hp_pct"] - bb["hp_pct"]
                types = tuple(sorted(ctx.stats.get(b["name"], {}).get("types", [])))
                base, _ = ctx.base(a["name"], b["name"], move["name"])
                rows.append(
                    (
                        view["group"],
                        move["name"],
                        a["name"],
                        b["name"],
                        types,
                        loss,
                        base,
                    )
                )
                if base > 0.02 and loss > 0:
                    ratios.append(loss / base)
    calibration = float(np.clip(np.median(ratios), 0.05, 1.0)) if ratios else 0.4
    return rows, calibration


class ResponseTable:
    def __init__(self, rows, calibration=None):
        ratios = [r[5] / r[6] for r in rows if r[6] > 0.02 and r[5] > 0]
        self.calibration = (
            float(np.clip(np.median(ratios), 0.05, 1.0)) if ratios else 0.4
        )
        self.tables = [defaultdict(lambda: [0.0, 0]) for _ in range(4)]
        for _, move, attacker, defender, types, loss, _ in rows:
            for table, key in zip(
                self.tables,
                [move, (move, attacker), (move, types), (move, attacker, defender)],
            ):
                table[key][0] += loss
                table[key][1] += 1

    @lru_cache(maxsize=100000)
    def estimate(self, move, attacker, defender, types):
        total, count = self.tables[0].get(move, (0.0, 0))
        if not count:
            return 0.0, 0.0
        mu = total / count
        a, na = self.tables[1].get((move, attacker), (0.0, 0))
        t, nt = self.tables[2].get((move, types), (0.0, 0))
        backoff = 0.5 * ((a + 20 * mu) / (na + 20) + (t + 20 * mu) / (nt + 20))
        pair, npair = self.tables[3].get((move, attacker, defender), (0.0, 0))
        return (pair + 15 * backoff) / (npair + 15), npair


class Context(TacticalContext):
    def fit(self, battles, views, chart, top):
        super().fit(battles, views, chart, top)
        self.sets = defaultdict(Counter)
        for view in views:
            for side in ["p1", "p2"]:
                for name, moves in view["moves"][side].items():
                    if 0 < len(moves) <= 4 and not moves & COPY:
                        self.sets[name][tuple(sorted(moves))] += 1
        self.set_cache = {}
        return self

    def hypotheses(self, name, known):
        key = name, tuple(sorted(known))
        if key in self.set_cache:
            return self.set_cache[key]
        known = frozenset(known)
        if len(known) >= 4 or known & COPY:
            out = [(known, 1.0)]
        else:
            options = defaultdict(float)
            for observed, count in self.sets.get(name, {}).items():
                observed = frozenset(observed)
                union = known | observed
                if len(union) <= 4:
                    # Partial observations are not complete sets. Prefer shared evidence.
                    options[union] += count * (0.2 ** len(known - observed))
            options[known] += 1.0
            selected = sorted(
                options.items(), key=lambda z: (-z[1], tuple(sorted(z[0])))
            )[:4]
            total = sum(w for _, w in selected)
            out = [(moves, w / total) for moves, w in selected]
        self.set_cache[key] = out
        return out


def hazards(battle, ctx):
    out = {}
    timeline = battle["battle_timeline"]
    for side, other in [("p1", "p2"), ("p2", "p1")]:
        chance = 0.0
        last_move, last_hp = {}, {}
        for i, turn in enumerate(timeline):
            st, enemy = turn[side + "_pokemon_state"], turn[other + "_pokemon_state"]
            move, enemy_move = (
                turn[side + "_move_details"],
                turn[other + "_move_details"],
            )
            previous = timeline[i - 1] if i else None
            fresh_enemy = bool(
                enemy_move and enemy_move != last_move.get(("enemy", enemy["name"]))
            )
            fresh_own = bool(move and move != last_move.get(("own", st["name"])))
            if fresh_enemy and enemy_move["name"] == "spikes":
                chance = 0.85
            same = bool(
                previous and previous[side + "_pokemon_state"]["name"] == st["name"]
            )
            if fresh_own and same and move["name"] == "rapidspin":
                if "ghost" not in ctx.stats.get(enemy["name"], {}).get("types", []):
                    chance = 0.15
            if previous and not same:
                grounded = "flying" not in ctx.stats.get(st["name"], {}).get(
                    "types", []
                )
                loss = last_hp.get(st["name"], 1.0) - st["hp_pct"]
                if grounded and (abs(loss - 0.125) < 0.02 or abs(loss - 0.0625) < 0.02):
                    chance = max(chance, 0.9)
            last_move["enemy", enemy["name"]] = enemy_move
            last_move["own", st["name"]] = move
            last_hp[st["name"]] = st["hp_pct"]
        out[side] = chance
    return out


def state(view, side, name):
    return view["states"][side].get(
        name, {"hp_pct": 1.0, "status": "nostatus", "boosts": {}, "effects": []}
    )


def alive(view, side, roster):
    return [n for n in roster if state(view, side, n)["status"] != "fnt"]


def remaining_count(view, side):
    return 6 - sum(st["status"] == "fnt" for st in view["states"][side].values())


def action_score(st, known):
    if st["status"] == "fnt":
        return 0.0
    if st["status"] == "frz":
        return 0.1
    if st["status"] == "slp":
        return 0.5 if "sleeptalk" in known else 0.15
    return 0.75 if st["status"] == "par" else 1.0


def accuracy(move):
    value = move.get("accuracy", 100)
    if isinstance(value, bool) or not value:
        return 1.0
    value = float(value)
    if FEATURE_FIXES and value <= 1:
        # The data stores accuracy as a fraction (1.0, 0.85, ...), not a percentage.
        return value
    return min(value / 100, 1.0)


class BattleMatrices:
    def __init__(self, ctx, table, battle, view, scenarios, modes=None, worlds=True):
        self.ctx, self.table, self.battle, self.view = ctx, table, battle, view
        modes = MODES if modes is None else modes
        self.active = {
            s: battle["battle_timeline"][-1][s + "_pokemon_state"]["name"]
            for s in ["p1", "p2"]
        }
        self.names = {
            "p1": sorted(view["team"]),
            "p2": sorted(set().union(*(set(r) for r, _ in scenarios))),
        }
        self.index = {
            s: {n: i for i, n in enumerate(ns)} for s, ns in self.names.items()
        }
        self.attack = {}
        self.known = {}
        self.boom = {}
        self.priority = {}
        self.worlds = {}
        for side, other in [("p1", "p2"), ("p2", "p1")]:
            shape = len(self.names[side]), len(self.names[other])
            self.attack[side] = {mode: np.zeros(shape) for mode in modes}
            self.known[side] = {mode: np.zeros(shape) for mode in modes}
            self.boom[side] = np.zeros(shape)
            self.priority[side] = np.zeros(shape)
            self.worlds[side] = np.zeros((4, *shape))
            for i, a in enumerate(self.names[side]):
                known = view["moves"][side].get(a, set())
                hypotheses = ctx.hypotheses(a, known)
                import hashlib

                offset = (
                    int(hashlib.sha256((side + ":" + a).encode()).hexdigest()[:8], 16)
                    / 2**32
                )
                cumulative = np.cumsum([w for _, w in hypotheses])
                draws = [
                    min(
                        int(np.searchsorted(cumulative, ((k + 0.5) / 4 + offset) % 1)),
                        len(hypotheses) - 1,
                    )
                    for k in range(4)
                ]
                sa = state(view, side, a)
                for j, b in enumerate(self.names[other]):
                    sb = state(view, other, b)
                    values, learned, accurate = {}, {}, {}
                    moves = set(known) | {m for m, _ in ctx.priors.get(a, [])}
                    moves |= set().union(*(set(ms) for ms, _ in hypotheses))
                    for m in moves:
                        md = ctx.moves.get(m, {})
                        value, cat = ctx.base(a, b, m)
                        if value:
                            atk, de = (
                                ("atk", "def") if cat == "physical" else ("spa", "spd")
                            )
                            if self.active[side] == a:
                                value *= stage(sa["boosts"].get(atk, 0))
                            if self.active[other] == b:
                                value /= stage(sb["boosts"].get(de, 0))
                            if sa["status"] == "brn" and cat == "physical":
                                value *= 0.5
                        values[m] = value
                        if (
                            md.get("category") not in {None, "status"}
                            and md.get("base_power", 0) <= 0
                        ):
                            types = tuple(sorted(ctx.stats.get(b, {}).get("types", [])))
                            response, _ = table.estimate(m, a, b, types)
                            if (
                                m
                                in {
                                    "return",
                                    "frustration",
                                    "present",
                                    "magnitude",
                                    "reversal",
                                    "flail",
                                }
                                and ctx.type_mult(md["type"], types) == 0
                            ):
                                response = 0.0
                            learned[m] = response / table.calibration
                        else:
                            learned[m] = value
                        accurate[m] = learned[m] * accuracy(md)
                    complete = len(known) >= 4 and not known & COPY
                    for mode in modes:
                        vals = (
                            learned
                            if mode in {"power", "corrected", "consistent"}
                            else values
                        )
                        remove_boom = mode in {"sustain", "corrected", "consistent"}
                        observed = max(
                            (
                                vals.get(m, 0.0)
                                for m in known
                                if not remove_boom or m not in BOOM
                            ),
                            default=0.0,
                        )
                        prior = max(
                            (
                                vals.get(m, 0.0) * p
                                for m, p in ctx.priors.get(a, [])
                                if not remove_boom or m not in BOOM
                            ),
                            default=0.0,
                        )
                        if complete and mode in {"slots", "corrected", "consistent"}:
                            prior = 0.0
                        if mode == "consistent":
                            total = sum(
                                w
                                * max(
                                    (accurate.get(m, 0.0) for m in ms if m not in BOOM),
                                    default=0.0,
                                )
                                for ms, w in hypotheses
                            )
                        else:
                            total = max(observed, prior)
                        self.attack[side][mode][i, j] = total
                        self.known[side][mode][i, j] = observed
                    self.boom[side][i, j] = sum(
                        w
                        * max(
                            (2 * accurate.get(m, 0.0) for m in ms if m in BOOM),
                            default=0.0,
                        )
                        for ms, w in hypotheses
                    )
                    if worlds:
                        for world, choice in enumerate(draws):
                            self.worlds[side][world, i, j] = max(
                                (
                                    accurate.get(m, 0.0)
                                    for m in hypotheses[choice][0]
                                    if m not in BOOM
                                ),
                                default=0.0,
                            )
                    best = max(
                        (
                            m
                            for m in sorted(known)
                            if m not in BOOM
                            and ctx.moves.get(m, {}).get("category")
                            not in {None, "status"}
                        ),
                        key=lambda m: accurate.get(m, 0.0),
                        default=None,
                    )
                    self.priority[side][i, j] = (
                        ctx.moves.get(best, {}).get("priority", 0) if best else 0
                    )

    def array(self, side, mode, mine, theirs, known=False, world=None):
        other = "p2" if side == "p1" else "p1"
        source = (
            self.known[side][mode]
            if known
            else (
                self.worlds[side][world]
                if world is not None
                else self.attack[side][mode]
            )
        )
        return source[
            np.ix_(
                [self.index[side][n] for n in mine],
                [self.index[other][n] for n in theirs],
            )
        ]


def threat_features(matrices, scenarios, mode, world=None):
    if mode == "consistent" and world is None:
        examples = [threat_features(matrices, scenarios, mode, k) for k in range(4)]
        return {key: sum(row[key] for row in examples) / 4 for key in examples[0]}
    view = matrices.view
    result = defaultdict(float)
    for roster, weight in scenarios:
        names = {"p1": alive(view, "p1", view["team"]), "p2": alive(view, "p2", roster)}
        local = {}
        for side, other in [("p1", "p2"), ("p2", "p1")]:
            mine, theirs = names[side], names[other]
            a = matrices.array(side, mode, mine, theirs, world=world)
            b = matrices.array(other, mode, theirs, mine, world=world).T
            hp1 = np.array([max(state(view, side, n)["hp_pct"], 0.05) for n in mine])
            hp2 = np.array([max(state(view, other, n)["hp_pct"], 0.05) for n in theirs])
            pressure = a / hp2[None, :]
            reply = b / hp1[:, None]
            known = matrices.array(side, mode, mine, theirs, known=True) / hp2[None, :]
            if mine and theirs:
                margins = np.log1p(pressure) - np.log1p(reply)
                worst = margins.min(axis=1)
                values = [
                    worst.max(),
                    worst.mean(),
                    (worst > 0).sum(),
                    margins.max(),
                    known.max(),
                    pressure.max(),
                    (
                        worst * np.array([state(view, side, n)["hp_pct"] for n in mine])
                    ).max(),
                ]
            else:
                values = [0.0] * 7
            for suffix, value in zip(
                [
                    "best_worst",
                    "mean_worst",
                    "positive_worst",
                    "favorable_pairs_max",
                    "known_pressure",
                    "expected_pressure",
                    "hp_weighted",
                ],
                values,
            ):
                local[side + "_threat_" + suffix] = float(value)
            local[side + "_threat_unknown_hp_moves"] = sum(
                "hiddenpower" in view["moves"][side].get(n, set()) for n in mine
            )
        for suffix in [
            "best_worst",
            "mean_worst",
            "positive_worst",
            "known_pressure",
            "expected_pressure",
            "hp_weighted",
        ]:
            local["diff_threat_" + suffix] = (
                local["p1_threat_" + suffix] - local["p2_threat_" + suffix]
            )
        for key, value in local.items():
            result[key] += weight * value
    return dict(result)


def identity_features(ctx, battle, view):
    f = {}
    for side in ["p1", "p2"]:
        st = battle["battle_timeline"][-1][side + "_pokemon_state"]
        known = view["moves"][side].get(st["name"], set())
        for name in ctx.top[:20]:
            f[f"{side}_identity_{name}"] = int(st["name"] == name)
        for status in STATUS:
            f[f"{side}_identity_status_{status}"] = int(st["status"] == status)
        for role, moves in [
            ("rest", {"rest"}),
            ("talk", {"sleeptalk"}),
            ("heal", HEAL),
            ("phaze", PHAZE),
            ("haze", {"haze"}),
            ("boom", BOOM),
            ("sleep", {"lovelykiss", "sleeppowder", "hypnosis", "spore", "sing"}),
        ]:
            f[f"{side}_identity_{role}"] = int(bool(known & moves))
        f[side + "_identity_moves_known"] = len(known)
        f[side + "_identity_unseen_species"] = int(st["name"] not in ctx.stats)
    return f


def speed(ctx, battle, view, side, name):
    base = ctx.stats.get(name, {}).get("base_spe", 50)
    level = (
        view["levels"].get(name, 100)
        if side == "p1"
        else (
            battle["p2_lead_details"]["level"]
            if name == battle["p2_lead_details"]["name"]
            else 100
        )
    )
    st = state(view, side, name)
    active = battle["battle_timeline"][-1][side + "_pokemon_state"]["name"] == name
    return (
        (2 * base + 100)
        * level
        / 100
        * (stage(st["boosts"].get("spe", 0)) if active else 1.0)
        * (0.25 if st["status"] == "par" else 1.0)
    )


def active_features(matrices):
    ctx, view, battle = matrices.ctx, matrices.view, matrices.battle
    f = {}
    speeds = {}
    for side, other in [("p1", "p2"), ("p2", "p1")]:
        name, enemy = matrices.active[side], matrices.active[other]
        st = state(view, side, name)
        known = view["moves"][side].get(name, set())
        i, j = matrices.index[side].get(name), matrices.index[other].get(enemy)
        damage = (
            matrices.attack[side]["consistent"][i, j] * matrices.table.calibration
            if i is not None and j is not None
            else 0.0
        )
        reply = (
            matrices.attack[other]["consistent"][j, i] * matrices.table.calibration
            if i is not None and j is not None
            else 0.0
        )
        speed_value = speed(ctx, battle, view, side, name)
        speeds[side] = speed_value
        pp = side + "_tempo_"
        f.update(
            {
                pp + "speed": speed_value,
                pp + "damage_index": damage,
                pp + "incoming_index": reply,
                pp + "survival_margin": st["hp_pct"] - reply,
                pp + "two_hit_margin": st["hp_pct"] - 2 * reply,
                pp + "action_score": action_score(st, known),
                pp + "recovery_known": int(bool(known & HEAL)),
                pp + "resttalk": int({"rest", "sleeptalk"} <= known),
                pp
                + "response_priority": (
                    float(matrices.priority[side][i, j])
                    if i is not None and j is not None
                    else 0.0
                ),
                pp + "moves_complete": int(len(known) >= 4 and not known & COPY),
                pp
                + "unknown_power_count": sum(
                    ctx.moves.get(m, {}).get("category") != "status"
                    and ctx.moves.get(m, {}).get("base_power", 0) <= 0
                    for m in known
                ),
            }
        )
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            f[pp + "base_" + stat] = ctx.stats.get(name, {}).get("base_" + stat, 0)
        f[pp + "phaze_last_mon"] = int(bool(known & PHAZE)) * int(
            remaining_count(view, other) <= 1
        )
        f[pp + "perish_trapped"] = int(
            {"perish", "trapped"} <= set(st.get("effects", []))
        )
    f["diff_tempo_log_speed"] = float(
        np.log(max(speeds["p1"], 1) / max(speeds["p2"], 1))
    )
    for side, other in [("p1", "p2"), ("p2", "p1")]:
        f[side + "_tempo_recovery_first"] = (
            f[side + "_tempo_recovery_known"]
            * float(speeds[side] > speeds[other])
            * f[side + "_tempo_survival_margin"]
        )
    return differences(f, "tempo_")


def response_graph(matrices, mine, theirs, side, hazard, risk):
    """Rows are attackers; columns are possible answers on the other team."""
    other = "p2" if side == "p1" else "p1"
    view = matrices.view
    a = np.clip(
        matrices.array(side, "consistent", mine, theirs) * matrices.table.calibration,
        0,
        1.5,
    )
    b = np.clip(
        matrices.array(other, "consistent", theirs, mine).T
        * matrices.table.calibration,
        0,
        1.5,
    )
    ha = np.array([state(view, side, n)["hp_pct"] for n in mine])
    hb = np.array([state(view, other, n)["hp_pct"] for n in theirs])
    aa = np.array(
        [
            action_score(state(view, side, n), view["moves"][side].get(n, set()))
            for n in mine
        ]
    )
    ab = np.array(
        [
            action_score(state(view, other, n), view["moves"][other].get(n, set()))
            for n in theirs
        ]
    )
    entry = np.array(
        [
            hazard
            * 0.125
            * ("flying" not in matrices.ctx.stats.get(n, {}).get("types", []))
            * (n != matrices.active[other])
            for n in theirs
        ]
    )
    onbench = np.array([n != matrices.active[other] for n in theirs])
    remaining = np.maximum(hb - entry, 0)
    survive_entry = sigmoid((remaining[None, :] - a * onbench[None, :]) / 0.08)
    survive_entry *= (remaining > 0)[None, :]
    incoming = a * aa[:, None]
    outgoing = b * ab[None, :]
    turns_def = remaining[None, :] / (incoming + 0.03)
    turns_atk = ha[:, None] / (outgoing + 0.03)
    sp_a = np.array([speed(matrices.ctx, matrices.battle, view, side, n) for n in mine])
    sp_b = np.array(
        [speed(matrices.ctx, matrices.battle, view, other, n) for n in theirs]
    )
    initiative = (sp_b[None, :] > sp_a[:, None]).astype(float) - 0.5
    duel = sigmoid((turns_def - turns_atk + initiative) * 1.3)
    duel *= np.minimum(1.0, outgoing / 0.05)
    recover = np.array(
        [
            (
                (0.3 if "rest" in view["moves"][other].get(n, set()) else 0.45)
                if view["moves"][other].get(n, set()) & HEAL
                else 0.0
            )
            for n in theirs
        ]
    )
    wall = (
        sigmoid((recover[None, :] * ab[None, :] - incoming) / 0.07)
        * (recover > 0)[None, :]
    )
    setup = np.array(
        [
            (
                max(
                    state(view, side, n)["boosts"].get("atk", 0),
                    state(view, side, n)["boosts"].get("spa", 0),
                    0,
                )
                if n == matrices.active[side]
                else 0.0
            )
            for n in mine
        ]
    )
    phaze = np.array(
        [
            bool(view["moves"][other].get(n, set()) & PHAZE)
            and remaining_count(view, side) > 1
            for n in theirs
        ],
        float,
    )
    haze = np.array(
        ["haze" in view["moves"][other].get(n, set()) for n in theirs], float
    )
    control = (
        (setup > 0)[:, None]
        * np.maximum(phaze, haze)[None, :]
        * sigmoid((remaining[None, :] - a) / 0.08)
        * ab[None, :]
    )
    answer = np.maximum.reduce([duel, wall, control]) * survive_entry
    keep = np.array([1 - risk.get(n, 0.0) if other == "p2" else 1.0 for n in theirs])
    return answer, answer * keep[None, :], entry, a, b


def graph_summary(answer, defender_hp):
    n, m = answer.shape
    if not n or not m:
        return {
            "unanswered": float(n),
            "best_unanswered": float(bool(n)),
            "mean_answer": 0.0,
            "single_answer": 0.0,
            "shared_load": 0.0,
            "fragility": 0.0,
            "backup": 0.0,
        }
    best = answer.max(axis=1)
    second = np.sort(answer, axis=1)[:, -2] if m > 1 else np.zeros(n)
    load = (
        answer / (answer.sum(axis=1, keepdims=True) + 0.01) * (1 - second[:, None])
    ).sum(axis=0)
    impacts = np.array(
        [
            np.sum(best - (np.delete(answer, j, axis=1).max(axis=1) if m > 1 else 0.0))
            for j in range(m)
        ]
    )
    return {
        "unanswered": float((1 - best).sum()),
        "best_unanswered": float(1 - best.min()),
        "mean_answer": float(best.mean()),
        "single_answer": float((best - second).sum()),
        "shared_load": float(np.max(load / (np.asarray(defender_hp) + 0.15))),
        "fragility": float(impacts.max()),
        "backup": float(second.mean()),
    }


def strategy_features(matrices, scenarios, hazard, risk, coverage_only=False):
    view = matrices.view
    out = {name: defaultdict(float) for name in ["coverage", "trades", "hazard"]}
    for roster, weight in scenarios:
        names = {"p1": alive(view, "p1", view["team"]), "p2": alive(view, "p2", roster)}
        graphs = {}
        for side, other in [("p1", "p2"), ("p2", "p1")]:
            mine, theirs = names[side], names[other]
            hp = [state(view, other, n)["hp_pct"] for n in theirs]
            graph, soft, entry, a, b = response_graph(
                matrices, mine, theirs, side, hazard[other], risk
            )
            graphs[side] = graph
            for key, value in graph_summary(graph, hp).items():
                out["coverage"][side + "_coverage_" + key] += weight * value
            for key, value in graph_summary(soft, hp).items():
                out["coverage"][side + "_coverage_risk_" + key] += weight * value
            if coverage_only:
                continue
            clean, _, _, _, _ = response_graph(matrices, mine, theirs, side, 0.0, risk)
            loss = (
                (clean.max(axis=1) - graph.max(axis=1)).sum()
                if mine and theirs
                else 0.0
            )
            out["hazard"][side + "_hazard_answer_loss"] += weight * float(loss)
            for key, value in summaries(entry).items():
                out["hazard"][other + "_hazard_entry_" + key] += weight * value
            out["hazard"][other + "_hazard_probability"] += weight * hazard[other]
            safe = [
                max(
                    0.0,
                    state(view, other, n)["hp_pct"]
                    - float(a[:, j].max() if mine else 0.0),
                )
                / (0.125 * hazard[other] + 0.03)
                for j, n in enumerate(theirs)
            ]
            out["hazard"][other + "_hazard_switch_budget_min"] += weight * float(
                min(safe, default=0.0)
            )
        if coverage_only:
            continue
        for side, other in [("p1", "p2"), ("p2", "p1")]:
            mine, theirs = names[side], names[other]
            g, reverse = graphs[side], graphs[other]
            if not mine or not theirs:
                vals = [0.0, 0.0, 0.0, 0.0]
            else:
                before = g.max(axis=1)
                own_before = reverse.max(axis=1)
                opened = np.stack(
                    [
                        before
                        - (
                            np.delete(g, j, axis=1).max(axis=1)
                            if len(theirs) > 1
                            else 0.0
                        )
                        for j in range(len(theirs))
                    ],
                    axis=1,
                )
                exposed = np.stack(
                    [
                        own_before
                        - (
                            np.delete(reverse, i, axis=1).max(axis=1)
                            if len(mine) > 1
                            else 0.0
                        )
                        for i in range(len(mine))
                    ],
                    axis=1,
                )
                # Both participants disappear in a successful trade.
                benefit = opened.sum(axis=0)[None, :] - opened
                cost = (exposed.sum(axis=0)[None, :] - exposed).T
                ix = np.ix_(
                    [matrices.index[side][n] for n in mine],
                    [matrices.index[other][n] for n in theirs],
                )
                boom = matrices.boom[side][ix] * matrices.table.calibration
                hp = np.array([state(view, other, n)["hp_pct"] for n in theirs])
                hit = sigmoid((boom - hp[None, :]) / 0.12) * (boom > 0)
                net = hit * (1 + benefit) - (1 + cost) * (boom > 0)
                vals = [
                    float(hit.max()),
                    float(net.max()),
                    float((net > 0).sum()),
                    float((hit * benefit).max()),
                ]
            for key, value in zip(
                ["ko_index", "net_value", "positive_options", "coverage_removed"], vals
            ):
                out["trades"][side + "_trade_" + key] += weight * value
    return {
        name: differences(
            dict(f),
            {"coverage": "coverage_", "trades": "trade_", "hazard": "hazard_"}[name],
        )
        for name, f in out.items()
    }


def prefix_view(battle, view, turns=20):
    # Use the final move/roster information for both resource states.
    b = dict(battle, battle_timeline=battle["battle_timeline"][:turns])
    v = dict(view)
    v["states"] = {
        side: {
            t[side + "_pokemon_state"]["name"]: t[side + "_pokemon_state"]
            for t in b["battle_timeline"]
        }
        for side in ["p1", "p2"]
    }
    return b, v
