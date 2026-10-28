"""State-dependent attack potential using only the supplied training observations.

Scores are relative pressure proxies, not exact damage or guaranteed victories.
Unknown-power moves remain unknown; no external movesets are introduced.
"""

from collections import Counter, defaultdict
import numpy as np
from .team_context import TeamContext, top_candidate_indices


def stage(s):
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


class TacticalContext:
    def fit(self, battles, views, chart, top_species):
        self.chart = chart
        self.top = top_species
        self.stats = {}
        self.moves = {}
        self.ms = Counter()
        self.ns = Counter()
        for b, v in zip(battles, views):
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                self.stats[p["name"]] = p
            for t in b["battle_timeline"]:
                for side in ["p1", "p2"]:
                    m = t[side + "_move_details"]
                    if m:
                        self.moves[m["name"]] = m
            for side in ["p1", "p2"]:
                for n, ms in sorted(v["moves"][side].items()):
                    if not ms:
                        continue
                    self.ns[n] += 1
                    for m in sorted(ms):
                        self.ms[n, m] += 1
        self.priors = {}
        for n in self.stats:
            mm = [
                (m, self.ms[n, m] / max(self.ns[n], 1))
                for m in self.moves
                if self.ms[n, m]
                and self.moves[m]["category"] != "status"
                and self.moves[m]["base_power"] > 0
            ]
            self.priors[n] = sorted(mm, key=lambda z: -z[1])[:6]
        self.teams = TeamContext(enriched=True, moves=True).fit(
            views, np.zeros(len(views))
        )
        self.static = {}
        return self

    def type_mult(self, ty, types):
        v = 1.0
        for t in types:
            v *= self.chart.get((ty, t), 1.0)
        return v

    def base(self, a, b, m):
        k = (a, b, m)
        if k in self.static:
            return self.static[k]
        A = self.stats.get(a)
        B = self.stats.get(b)
        M = self.moves.get(m)
        if not A or not B or not M or M["category"] == "status" or M["base_power"] <= 0:
            return (0.0, "")
        cat = M["category"]
        atk = "atk" if cat == "physical" else "spa"
        de = "def" if cat == "physical" else "spd"
        power = (
            M["base_power"]
            * self.type_mult(M["type"], B["types"])
            * (1.5 if M["type"] in A["types"] else 1.0)
        )
        value = (
            power
            * (2 * A["base_" + atk] + 100)
            / (2 * B["base_" + de] + 100)
            / (2 * B["base_hp"] + 200)
        )
        self.static[k] = (value, cat)
        return self.static[k]

    def attack(self, a, b, sa, sb, known, active_a, active_b):
        def value(m):
            v, cat = self.base(a, b, m)
            if not v:
                return 0.0
            atk = "atk" if cat == "physical" else "spa"
            de = "def" if cat == "physical" else "spd"
            if active_a:
                v *= stage(sa.get("boosts", {}).get(atk, 0))
            if active_b:
                v /= stage(sb.get("boosts", {}).get(de, 0))
            if sa.get("status") == "brn" and cat == "physical":
                v *= 0.5
            return v

        observed = max((value(m) for m in known), default=0.0)
        prior = max((value(m) * p for m, p in self.priors.get(a, [])), default=0.0)
        return observed, max(observed, prior)

    def rosters(self, v):
        ca = self.teams.candidates(v)
        w = self.teams.weights(v, ca)
        if not ca:
            return [(tuple(sorted(v["seen"])), 1.0)], ca, w
        ix = top_candidate_indices(w, 3)
        mass = w[ix].sum()
        return [(ca[i], float(w[i] / mass)) for i in ix], ca, w

    def threat_features(self, b, v):
        scenarios, _, _ = self.rosters(v)
        result = defaultdict(float)
        active = {
            s: b["battle_timeline"][-1][s + "_pokemon_state"]["name"]
            for s in ["p1", "p2"]
        }
        for opteam, weight in scenarios:
            roster = {"p1": v["team"], "p2": opteam}
            alive = {}
            for side in ["p1", "p2"]:
                alive[side] = {}
                for n in roster[side]:
                    st = v["states"][side].get(
                        n, {"hp_pct": 1.0, "status": "nostatus", "boosts": {}}
                    )
                    if st["status"] != "fnt":
                        alive[side][n] = st
            local = {}
            for me, op in [("p1", "p2"), ("p2", "p1")]:
                best = []
                worst = []
                hpweight = []
                knownbest = []
                exposed = []
                for a, sa in alive[me].items():
                    margins = []
                    knowns = []
                    threats = []
                    for bb, sb in alive[op].items():
                        ka, pa = self.attack(
                            a,
                            bb,
                            sa,
                            sb,
                            v["moves"][me].get(a, []),
                            active[me] == a,
                            active[op] == bb,
                        )
                        kb, pb = self.attack(
                            bb,
                            a,
                            sb,
                            sa,
                            v["moves"][op].get(bb, []),
                            active[op] == bb,
                            active[me] == a,
                        )
                        hp_a = max(sa["hp_pct"], 0.05)
                        hp_b = max(sb["hp_pct"], 0.05)
                        pressure = pa / hp_b
                        reply = pb / hp_a
                        margins.append(float(np.log1p(pressure) - np.log1p(reply)))
                        threats.append(pressure)
                        knowns.append(ka / hp_b)
                    if margins:
                        worst.append(min(margins))
                        best.append(max(margins))
                        hpweight.append(sa["hp_pct"] * min(margins))
                        knownbest.append(max(knowns))
                        exposed.append(max(threats))
                pre = me + "_threat_"
                local[pre + "best_worst"] = max(worst, default=0.0)
                local[pre + "mean_worst"] = float(np.mean(worst)) if worst else 0.0
                local[pre + "positive_worst"] = sum(x > 0 for x in worst)
                local[pre + "favorable_pairs_max"] = max(best, default=0.0)
                local[pre + "known_pressure"] = max(knownbest, default=0.0)
                local[pre + "expected_pressure"] = max(exposed, default=0.0)
                local[pre + "hp_weighted"] = max(hpweight, default=0.0)
                local[pre + "unknown_hp_moves"] = sum(
                    "hiddenpower" in v["moves"][me].get(n, []) for n in alive[me]
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
            for k, val in local.items():
                result[k] += weight * val
        return dict(result)
