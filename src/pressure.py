"""Estimate a pressure proxy for attacks whose supplied base power is zero."""

import numpy as np
import pandas as pd
from . import state as deep
from .tactics import TacticalContext, stage


def damage_samples(B, ctx):
    """New move tokens with the same pair of active species and uncensored HP loss."""
    rows = []
    for bi, b in enumerate(B):
        last = {"p1": {}, "p2": {}}
        tl = b["battle_timeline"]
        for i, t in enumerate(tl):
            for side, opp in [("p1", "p2"), ("p2", "p1")]:
                a = t[side + "_pokemon_state"]
                d = t[opp + "_pokemon_state"]
                m = t[side + "_move_details"]
                old_m = last[side].get(a["name"])
                if m:
                    last[side][a["name"]] = m
                if not i or not m or m == old_m or m["category"] == "status":
                    continue
                a0 = tl[i - 1][side + "_pokemon_state"]
                d0 = tl[i - 1][opp + "_pokemon_state"]
                if a0["name"] != a["name"] or d0["name"] != d["name"]:
                    continue
                if (
                    d["status"] != "nostatus"
                    or d0["status"] != "nostatus"
                    or d["hp_pct"] <= 0.05
                ):
                    continue
                if any(
                    e in set(d["effects"]) | set(d0["effects"])
                    for e in ["substitute", "leechseed", "perish"]
                ):
                    continue
                if (
                    a["status"] in ["slp", "frz", "fnt"]
                    or a0["boosts"] != a["boosts"]
                    or d0["boosts"] != d["boosts"]
                ):
                    continue
                dm = t[opp + "_move_details"]
                if dm and dm["name"] in deep.RECOVERY:
                    continue
                loss = d0["hp_pct"] - d["hp_pct"]
                if not 0.01 <= loss <= 0.8:
                    continue
                A, D = ctx.stats.get(a["name"]), ctx.stats.get(d["name"])
                if A is None or D is None:
                    continue
                at, de = (
                    ("atk", "def") if m["category"] == "physical" else ("spa", "spd")
                )
                unit = (
                    (2 * A["base_" + at] + 100)
                    / (2 * D["base_" + de] + 100)
                    / (2 * D["base_hp"] + 200)
                )
                # The supplied Hidden Power type is not an observed type choice.
                if m["name"] != "hiddenpower":
                    unit *= ctx.type_mult(m["type"], D["types"]) * (
                        1.5 if m["type"] in A["types"] else 1.0
                    )
                unit *= stage(a0["boosts"].get(at, 0)) / stage(d0["boosts"].get(de, 0))
                if a0["status"] == "brn" and at == "atk":
                    unit *= 0.5
                if unit > 0:
                    rows.append(
                        (bi, a["name"], m["name"], m["base_power"], loss / unit)
                    )
    return pd.DataFrame(
        rows, columns=["battle", "attacker", "move", "known_power", "effective_power"]
    )


def fit_power(samples, indices):
    S = samples[samples.battle.isin(indices)]
    known = S[S.known_power > 0]
    scale = (
        float(np.median(known.effective_power / known.known_power))
        if len(known)
        else 1.0
    )
    scale = max(scale, 1e-6)
    unknown = S[S.known_power == 0]
    pooled, specific = {}, {}
    for move, g in unknown.groupby("move", sort=True):
        if len(g) >= 5:
            pooled[move] = float(np.clip(np.median(g.effective_power) / scale, 1, 250))
    for (a, move), g in unknown.groupby(["attacker", "move"], sort=True):
        if move in pooled:
            value = float(np.clip(np.median(g.effective_power) / scale, 1, 250))
            specific[a, move] = (len(g) * value + 20 * pooled[move]) / (len(g) + 20)
    return dict(
        pooled=pooled,
        specific=specific,
        sample_count=len(S),
        unknown_count=len(unknown),
        scale=scale,
    )


class EmpiricalTactics(TacticalContext):
    @classmethod
    def from_context(cls, original, fitted):
        obj = cls()
        obj.__dict__ = dict(original.__dict__)
        obj.static = {}
        obj.fitted_power = fitted
        obj.priors = {}
        for n in obj.stats:
            entries = [
                (m, obj.ms[n, m] / max(obj.ns[n], 1))
                for m in obj.moves
                if obj.ms[n, m]
                and obj.moves[m]["category"] != "status"
                and obj.power(n, m) > 0
            ]
            obj.priors[n] = sorted(entries, key=lambda z: -z[1])[:6]
        return obj

    def power(self, a, m):
        move = self.moves[m]
        if move["base_power"] > 0:
            return move["base_power"]
        return self.fitted_power["specific"].get(
            (a, m), self.fitted_power["pooled"].get(m, 0.0)
        )

    def base(self, a, b, m):
        k = a, b, m
        if k in self.static:
            return self.static[k]
        A, B, M = self.stats.get(a), self.stats.get(b), self.moves.get(m)
        if not A or not B or not M or M["category"] == "status":
            return 0.0, ""
        if M["base_power"] > 0:
            return super().base(a, b, m)
        power = self.power(a, m)
        if not power:
            return 0.0, ""
        cat = M["category"]
        at, de = ("atk", "def") if cat == "physical" else ("spa", "spd")
        if m != "hiddenpower":
            power *= self.type_mult(M["type"], B["types"]) * (
                1.5 if M["type"] in A["types"] else 1.0
            )
        value = (
            power
            * (2 * A["base_" + at] + 100)
            / (2 * B["base_" + de] + 100)
            / (2 * B["base_hp"] + 200)
        )
        self.static[k] = value, cat
        return self.static[k]
