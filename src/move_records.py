"""Smoothed outcome records for revealed species and move combinations."""

from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from .team_context import TeamContext, top_candidate_indices


class MoveRecords:
    def fit(self, views, y):
        self.ctx = TeamContext(enriched=True, moves=True).fit(views, y)
        self.g = Counter()
        self.w = defaultdict(float)
        self.pg = Counter()
        self.pw = defaultdict(float)
        self.tg = Counter()
        self.tw = defaultdict(float)
        for v, z in zip(views, y):
            ca = self.ctx.candidates(v)
            op = ca[0] if len(ca) == 1 else None
            for side, team, label in [
                ("p1", v["team"], float(z)),
                ("p2", op, 1 - float(z)),
            ]:
                for n, ms in sorted(v["moves"][side].items()):
                    for m in sorted(ms):
                        self.g[n, m] += 1
                        self.w[n, m] += label
                        if team is not None:
                            self.tg[team, n, m] += 1
                            self.tw[team, n, m] += label
                    for a, b in combinations(sorted(ms), 2):
                        self.pg[n, a, b] += 1
                        self.pw[n, a, b] += label
        return self

    def transform(self, views):
        rows = []
        for v in views:
            out = {}
            for side in ["p1", "p2"]:
                pre = side + "_setrecord_"
                gr = []
                pairs = []
                gs = []
                hp = []
                tokens = []
                for n, ms in sorted(v["moves"][side].items()):
                    st = v["states"][side].get(n, {})
                    h = 0.0 if st.get("status") == "fnt" else st.get("hp_pct", 1.0)
                    for m in sorted(ms):
                        gr.append((self.w[n, m] + 10) / (self.g[n, m] + 20))
                        gs.append(np.log1p(self.g[n, m]))
                        hp.append(h)
                        tokens.append((n, m))
                    for a, b in combinations(sorted(ms), 2):
                        pairs.append((self.pw[n, a, b] + 10) / (self.pg[n, a, b] + 20))
                gr = np.array(gr)
                hp = np.array(hp)
                out[pre + "global_mean"] = float(gr.mean()) if len(gr) else 0.5
                out[pre + "global_min"] = float(gr.min()) if len(gr) else 0.5
                out[pre + "global_max"] = float(gr.max()) if len(gr) else 0.5
                out[pre + "global_spread"] = float(gr.std()) if len(gr) else 0.0
                out[pre + "global_support"] = float(np.mean(gs)) if gs else 0.0
                out[pre + "global_alive"] = (
                    float(hp @ (gr - 0.5) / max(hp.sum(), 1e-6)) if len(gr) else 0.0
                )
                out[pre + "pair_mean"] = float(np.mean(pairs)) if pairs else 0.5
                out[pre + "pair_min"] = min(pairs, default=0.5)
                out[pre + "pair_max"] = max(pairs, default=0.5)
                if side == "p1":
                    ca = [v["team"]]
                    ww = np.array([1.0])
                else:
                    ca = self.ctx.candidates(v)
                    ww = self.ctx.weights(v, ca)
                    if len(ca) > 3:
                        ix = top_candidate_indices(ww, 3)
                        ca = [ca[i] for i in ix]
                        ww = ww[ix] / ww[ix].sum()
                sums = np.zeros(6)
                for team, w in zip(ca, ww):
                    prior = self.ctx.rate(team)
                    rr = np.array(
                        [
                            (self.tw[team, n, m] + 20 * prior)
                            / (self.tg[team, n, m] + 20)
                            for n, m in tokens
                        ]
                    )
                    support = [self.tg[team, n, m] for n, m in tokens]
                    values = [
                        rr.mean() if len(rr) else prior,
                        rr.min() if len(rr) else prior,
                        rr.max() if len(rr) else prior,
                        np.mean(np.log1p(support)) if support else 0.0,
                        (
                            float(hp @ (rr - prior) / max(hp.sum(), 1e-6))
                            if len(rr)
                            else 0.0
                        ),
                        sum(g >= 2 for g in support),
                    ]
                    sums += w * np.array(values)
                if not ca:
                    sums[:3] = 0.5
                for name, val in zip(
                    [
                        "team_mean",
                        "team_min",
                        "team_max",
                        "team_support",
                        "team_alive_evidence",
                        "team_supported_tokens",
                    ],
                    sums,
                ):
                    out[pre + name] = val
            for key in list(out):
                if key.startswith("p1_"):
                    out["diff_" + key[3:]] = (
                        out[key] - out[key.replace("p1_", "p2_", 1)]
                    )
            rows.append(out)
        return pd.DataFrame(rows).astype(np.float32)
