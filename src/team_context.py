"""Smoothed team records and uncertain opponent rosters."""

from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def top_candidate_indices(weights, count=3):
    """Break numerically tied posterior weights by canonical roster order.

    Candidate rosters are sorted in candidates(). Rounding only the ranking
    key prevents floating-point accumulation noise from selecting another
    roster at the cutoff; the original probabilities retain their values.
    """
    weights = np.asarray(weights, float)
    return np.lexsort((np.arange(len(weights)), -np.round(weights, 12)))[:count]


class TeamContext:
    """Smoothed win records of known teams, and P2 roster matching.

    Known teams are P1 teams and, when enriched, P2 teams with all six revealed.
    A P2 team is matched to the known teams that contain every revealed P2
    species; with moves, candidates are reweighted by how often each one was
    seen with the revealed moves. Records are (wins + 5) / (games + 10).
    """

    def __init__(self, enriched=False, moves=False):
        self.enriched = enriched
        self.use_moves = moves

    def candidates(self, v):
        """Known teams that contain every revealed P2 species."""
        if len(v["seen"]) > 6:
            return []
        sets = [self.by.get(n, set()) for n in v["seen"]]
        return sorted(set.intersection(*sets)) if sets else []

    def fit(self, views, y):
        self.count = Counter(v["team"] for v in views)
        if self.enriched:
            self.count.update(
                tuple(sorted(v["seen"])) for v in views if len(v["seen"]) == 6
            )
        self.by = defaultdict(set)
        for k in self.count:
            for n in k:
                self.by[n].add(k)
        self.games = Counter()
        self.wins = defaultdict(float)
        self.ms = Counter()
        self.mt = Counter()
        self.gs = Counter()
        self.gt = Counter()
        for v, z in zip(views, y):
            k = v["team"]
            self.games[k] += 1
            self.wins[k] += float(z)
            ca = self.candidates(v)
            op = ca[0] if len(ca) == 1 else None
            if op is not None:
                self.games[op] += 1
                self.wins[op] += 1 - float(z)
            for side, tk in [("p1", k), ("p2", op)]:
                for n, ms in sorted(v["moves"][side].items()):
                    if not ms:
                        continue
                    self.gs[n] += 1
                    if tk is not None:
                        self.ms[tk, n] += 1
                    for m in sorted(ms):
                        self.gt[n, m] += 1
                        if tk is not None:
                            self.mt[tk, n, m] += 1
        return self

    def weights(self, v, ca):
        """Probability of each candidate team, from its count and the revealed moves."""
        if not ca:
            return np.array([])
        log = np.log(np.array([self.count[k] for k in ca], float))
        if self.use_moves:
            for n, ms in sorted(v["moves"]["p2"].items()):
                for m in sorted(ms):
                    prior = (self.gt[n, m] + 1) / (self.gs[n] + 4)
                    log += np.log(
                        [
                            (self.mt[k, n, m] + 8 * prior) / (self.ms[k, n] + 8)
                            for k in ca
                        ]
                    )
        log -= log.max()
        w = np.exp(log)
        return w / w.sum()

    def rate(self, k):
        """Smoothed win rate of team k: five wins in ten games before its own."""
        return (self.wins.get(k, 0) + 5) / (self.games.get(k, 0) + 10)

    def transform(self, views, mode="base"):
        """Record features; "posterior" adds the spread over the P2 candidates."""
        result = []
        for v in views:
            k = v["team"]
            r1 = self.rate(k)
            g1 = self.games.get(k, 0)
            ca = self.candidates(v)
            w = self.weights(v, ca)
            rates = np.array([self.rate(q) for q in ca])
            g = np.array([self.games.get(q, 0) for q in ca])
            best = int(top_candidate_indices(w, 1)[0]) if len(w) else -1
            r2 = float(rates[best]) if len(w) else 0.5
            g2 = float(g[best]) if len(w) else 0.0
            out = {
                "p1_team_record": r1,
                "p1_team_battles": g1,
                "p2_team_record": r2,
                "p2_team_battles": g2,
                "diff_team_record": r1 - r2,
                "p2_team_candidates": len(ca),
            }
            if mode != "base":
                mu = float(w @ rates) if len(w) else 0.5
                out.update(
                    p2_record_mean=mu,
                    p2_record_sd=(
                        float(np.sqrt(w @ (rates - mu) ** 2)) if len(w) else 0.0
                    ),
                    p2_record_min=float(rates.min()) if len(w) else 0.5,
                    p2_record_max=float(rates.max()) if len(w) else 0.5,
                    p2_match_mass=float(w.max()) if len(w) else 0.0,
                    p2_match_entropy=(
                        float(-(w * np.log(w + 1e-15)).sum()) if len(w) else 0.0
                    ),
                    p2_expected_support=float(w @ g) if len(w) else 0.0,
                    diff_expected_record=r1 - mu,
                    p2_six_known=int(len(v["seen"]) == 6),
                )
            result.append(out)
        return pd.DataFrame(result).astype(float)
