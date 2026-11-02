"""
Species co-occurrence, functional substitutability and soft roster completion.

No external lookup tables or handcrafted stat thresholds:
Roles and substitutes are derived purely from linear algebra on the team
co-occurrence matrix (Gram matrix of team partner profiles).

Key representations:
1. Co-occurrence matrix C: C[a, b] = number of teams containing both species a and b.
2. Bayesian conditional probability P(B in team | A in team).
3. Functional substitutability matrix S = C_norm @ C_norm.T:
   Two species with high cosine similarity between their partner profiles share
   the same functional team slot (e.g. Cloyster <-> Forretress as Spiker/Spinner,
   Blissey <-> Miltank as Cleric/Special Sponge), regardless of elemental types.
4. Latent team archetype embedding: projecting seen species into the partner space.
5. Bayesian posterior ranking of unrevealed bench species given visible members.

All statistics are strictly fitted only on the outer training fold.
"""

from collections import Counter
import numpy as np


class TeamCooccurrence:
    """
    Learns data-driven species co-occurrence and latent role substitutability
    from training team rosters.
    """

    def __init__(self, laplace_alpha=1.0):
        self.laplace_alpha = laplace_alpha
        self.team_count = 0
        self.species_counts = Counter()
        self.pair_counts = Counter()
        self.all_species = []
        self.species_to_idx = {}
        self.C = None            # Raw co-occurrence matrix (M, M)
        self.C_norm = None       # Row-normalized partner profile matrix (M, M)
        self.S = None            # Functional substitutability matrix (M, M)
        self.cond_probs = None   # P(B in team | A in team) (M, M)

    def fit(self, views):
        """
        Fit strictly on the provided training views (outer fold).
        Uses unique P1 teams + confirmed 6-species P2 views.
        No external stats, labels, or thresholds are used.
        """
        teams = set()
        for v in views:
            teams.add(v["team"])
            if len(v.get("seen", [])) == 6:
                teams.add(tuple(sorted(v["seen"])))

        self.team_count = len(teams)
        self.species_counts = Counter()
        self.pair_counts = Counter()

        for t in teams:
            for s in t:
                self.species_counts[s] += 1
            for a in t:
                for b in t:
                    if a != b:
                        self.pair_counts[(a, b)] += 1

        self.all_species = sorted(list(self.species_counts.keys()))
        self.species_to_idx = {s: i for i, s in enumerate(self.all_species)}
        M = len(self.all_species)

        # 1. Raw co-occurrence matrix C
        self.C = np.zeros((M, M), dtype=np.float32)
        for (a, b), count in self.pair_counts.items():
            if a in self.species_to_idx and b in self.species_to_idx:
                self.C[self.species_to_idx[a], self.species_to_idx[b]] = float(count)

        # 2. Partner profile normalization and Functional Substitutability Matrix S:
        # Cosine similarity between partner distributions C[a, :] and C[b, :]
        row_norms = np.linalg.norm(self.C, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        self.C_norm = self.C / row_norms
        self.S = self.C_norm @ self.C_norm.T  # Gram matrix of partner profiles

        # 3. Conditional probability with Laplace smoothing:
        # P(B | A) = (Count(A, B) + alpha) / (Count(A) + alpha * M)
        self.cond_probs = np.zeros((M, M), dtype=np.float32)
        for i, a in enumerate(self.all_species):
            c_a = self.species_counts[a]
            denom = c_a + self.laplace_alpha * M
            for j, b in enumerate(self.all_species):
                if i == j:
                    self.cond_probs[i, j] = 1.0
                else:
                    c_ab = self.C[i, j]
                    self.cond_probs[i, j] = (c_ab + self.laplace_alpha) / denom

        return self

    def substitutes(self, species_name, top_k=5):
        """
        Return the data-driven functional substitutes for a species,
        ranked by cosine similarity of their teambuilding contexts.
        """
        if species_name not in self.species_to_idx:
            return []
        i = self.species_to_idx[species_name]
        sims = [
            (self.all_species[j], float(self.S[i, j]))
            for j in range(len(self.all_species))
            if j != i
        ]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def team_embedding(self, seen_species):
        """
        Compute a compact latent archetype vector for the seen team members
        by averaging their normalized partner profiles.
        """
        indices = [self.species_to_idx[s] for s in seen_species if s in self.species_to_idx]
        if not indices:
            return np.zeros(len(self.all_species), dtype=np.float32)
        return np.mean(self.C_norm[indices], axis=0)

    def predict_unseen_species(self, seen_species, top_k=6):
        """
        Given the seen species of P2, rank all unseen candidate species by
        their log posterior probability under Naive Bayes:
        log P(B | seen) = log P(B) + sum_{s in seen} log [ P(B | s) / P(B) ]
        """
        seen_set = set(seen_species)
        candidates = [s for s in self.all_species if s not in seen_set]
        if not candidates or not seen_set:
            return []

        M = len(self.all_species)
        total_teams = max(self.team_count, 1)

        scores = []
        for cand in candidates:
            j = self.species_to_idx[cand]
            p_prior = (self.species_counts[cand] + self.laplace_alpha) / (
                total_teams + self.laplace_alpha * M
            )
            log_score = np.log(p_prior)

            for s in seen_set:
                if s in self.species_to_idx:
                    i = self.species_to_idx[s]
                    p_cond = self.cond_probs[i, j]
                    # Pointwise Mutual Information (PMI) boost: P(B|A) / P(B)
                    log_score += np.log(max(p_cond / p_prior, 1e-4))

            scores.append((cand, float(log_score)))

        scores.sort(key=lambda x: -x[1])
        log_vals = np.array([s[1] for s in scores])
        log_vals -= np.max(log_vals)
        probs = np.exp(log_vals) / np.sum(np.exp(log_vals))

        return [(cand, float(prob)) for (cand, _), prob in zip(scores, probs)][:top_k]


if __name__ == '__main__':
    import json
    from pathlib import Path
    data_path = Path('data/train.jsonl')
    if data_path.exists():
        print('Loading battles from data/train.jsonl...')
        views = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                b = json.loads(line)
                team = tuple(sorted(p['name'] for p in b['p1_team_details']))
                seen = sorted(set(t['p2_pokemon_state']['name'] for t in b['battle_timeline'] if t['p2_pokemon_state']['name']))
                views.append({'team': team, 'seen': seen})
        print(f'Fitting TeamCooccurrence on {len(views)} views...')
        tc = TeamCooccurrence().fit(views)
        print(f'Total unique species: {len(tc.all_species)}')
        for s in ['cloyster', 'snorlax', 'blissey', 'skarmory', 'gengar']:
            if s in tc.species_to_idx:
                subs = tc.substitutes(s, top_k=3)
                print(f'  Substitutes for {s:10s}: {subs}')
