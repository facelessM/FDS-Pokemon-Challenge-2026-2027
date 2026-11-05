"""Second stage: a small logistic regression on top of the tree models.

After the five-view model, a residual scan over a few hundred candidate
features found two signals that the trees still underused:

* Similar-team records. The win rate of teams that share most species with
  P1's team, and of known teams that contain every Pokémon P2 has revealed.
  Exact team matches are rare, so near matches carry most of the evidence.
* Spikes at turn 30. Whether Spikes is still down on each side, after Rapid
  Spin removals. Every later switch-in on that side loses HP.

The logistic regression takes logit(tree probability) and these signals. It is
trained on the out-of-fold probabilities of all three CV partitions, so it never
sees a probability that was fitted on its own battle.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from .observations import load
from .validation import N_FOLDS, SEEDS, THRESHOLD

PRIOR = 5.0  # pseudo-battles pulling a similar-team rate toward 0.5


# Similar-team records


def p1_team(battle):
    return {m["name"] for m in battle["p1_team_details"]}


def p2_revealed(battle):
    return {t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"]}


class SimilarTeams:
    """Teams as species indicator vectors. The vocabulary comes from training battles."""

    def __init__(self, train, test):
        names = sorted(set().union(*(p1_team(b) | p2_revealed(b) for b in train)))
        self.vocab = {n: i for i, n in enumerate(names)}
        battles = train + test
        self.p1 = self.onehot([p1_team(b) for b in battles])
        self.p2 = self.onehot([p2_revealed(b) for b in battles])
        self.revealed = self.p2.sum(1)
        self.y = None

    def onehot(self, teams):
        out = np.zeros((len(teams), len(self.vocab)), np.float32)
        for i, team in enumerate(teams):
            for name in team:
                if name in self.vocab:
                    out[i, self.vocab[name]] = 1
        return out

    def rates(self, fit, query):
        """Similar-team win rates for query rows, from the battles in fit only.

        Known teams are every P1 team (result y) and every P2 team with all six
        Pokémon revealed (result 1 - y). P1 weights a known team by 4^(shared - 6)
        when it shares at least four species. P2 weights it by 4^(-missing), where
        missing counts revealed P2 species absent from the known team (at most one).
        """
        six = fit[self.revealed[fit] == 6]
        teams = np.vstack([self.p1[fit], self.p2[six]])
        won = np.concatenate([self.y[fit], 1 - self.y[six]]).astype(np.float32)
        out = []
        for start in range(0, len(query), 2000):
            q = query[start : start + 2000]
            shared = self.p1[q] @ teams.T
            w1 = (4.0 ** (shared - 6)) * (shared >= 4)
            seen = self.revealed[q][:, None]
            missing = seen - self.p2[q] @ teams.T
            w2 = (4.0 ** (-missing)) * ((missing <= 1) & (seen >= 3))
            rate = lambda w: ((w * won).sum(1) + PRIOR / 2) / (w.sum(1) + PRIOR)
            out.append(np.column_stack([rate(w1), rate(w2)]))
        return np.vstack(out)


def inner_folds(y, groups, rows):
    """The same inner split the team records use."""
    split = StratifiedGroupKFold(5, shuffle=True, random_state=99)
    return [rows[a] for a, _ in split.split(np.zeros(len(rows)), y[rows], groups[rows])]


def similar_team_features(sim, inputs, test_rows):
    """Out-of-fold rows per partition, and test rows averaged over every fold.

    A validation row averages five inner models fitted on its outer training
    fold, so its rate never includes its own battle or its timeline group.
    """
    y, groups = inputs["y"], inputs["groups"]
    sim.y = y
    oof, test = [], []
    for seed in SEEDS:
        part = np.zeros((len(y), 2), np.float32)
        for fold in range(N_FOLDS):
            tr, va = inputs["partitions"][seed][fold]
            fits = inner_folds(y, groups, tr)
            part[va] = sum(sim.rates(f, va) for f in fits) / len(fits)
            test += [sim.rates(f, test_rows) for f in fits]
        oof.append(part)
    return oof, np.mean(test, axis=0)


# Spikes at turn 30


def spikes_difference(battle, types):
    """Spikes still down on P2's side minus P1's side after the 30 turns.

    Spikes goes down on the opponent's side. Rapid Spin clears the user's side
    unless the target is a Ghost, and only when the user was already in on the
    previous turn (a Pokémon that just switched in did not use the move).
    """
    timeline = battle["battle_timeline"]
    down = {"p1": 0, "p2": 0}
    for i, turn in enumerate(timeline):
        for side, other in (("p1", "p2"), ("p2", "p1")):
            move = turn[f"{side}_move_details"]
            if not move:
                continue
            if move["name"] == "spikes":
                down[other] = 1
            user = turn[f"{side}_pokemon_state"]["name"]
            target = turn[f"{other}_pokemon_state"]["name"]
            if (
                move["name"] == "rapidspin"
                and "ghost" not in types.get(target, [])
                and i > 0
                and timeline[i - 1][f"{side}_pokemon_state"]["name"] == user
            ):
                down[side] = 0
    return down["p2"] - down["p1"]


def spikes_feature(battles):
    types = {}
    for b in battles:
        for m in b["p1_team_details"]:
            types[m["name"]] = m["types"]
        types[b["p2_lead_details"]["name"]] = b["p2_lead_details"]["types"]
    return np.array([spikes_difference(b, types) for b in battles], float)


# Fit and apply


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(data, output, view, signals, name):
    """Fit the second stage on a finished predict run and write a submission.

    view: "blend" uses the equal mean of the five views, "tactical" only the
    tactical view. signals: "team" or "team,spikes".
    """
    from .pipeline import read_pickle

    inputs = read_pickle(output / "inputs.pkl.gz")
    oof = np.load(output / "oof.npz")
    test = np.load(output / "test_probabilities.npz")
    train_battles = load(data / "train.jsonl")
    test_battles = load(data / "test.jsonl")
    assert [b["battle_id"] for b in test_battles] == test["battle_id"].tolist()
    y = inputs["y"]

    if view == "blend":
        base_oof = [c.mean(1) for c in oof["components"]]
        base_test = test["p"]
    else:
        base_oof = [c[:, 2] for c in oof["components"]]
        base_test = test["components"][:, :, 2].mean(0)

    sim = SimilarTeams(train_battles, test_battles)
    test_rows = np.arange(len(y), len(y) + len(test_battles))
    team_oof, team_test = similar_team_features(sim, inputs, test_rows)
    columns = [[logit(b), t] for b, t in zip(base_oof, team_oof)]
    test_columns = [logit(base_test), team_test]
    if "spikes" in signals:
        spikes_train = spikes_feature(train_battles)
        spikes_test = spikes_feature(test_battles)
        columns = [c + [spikes_train] for c in columns]
        test_columns.append(spikes_test)
    X = [np.column_stack(c) for c in columns]

    # Report: each partition's rows scored by a model that did not see them.
    probabilities = []
    for x in X:
        p = np.zeros(len(y))
        split = StratifiedGroupKFold(5, shuffle=True, random_state=7)
        for a, b in split.split(x, y, inputs["groups"]):
            model = LogisticRegression(C=1.0, max_iter=2000).fit(x[a], y[a])
            p[b] = model.predict_proba(x[b])[:, 1]
        probabilities.append(p)
    averaged = np.mean(probabilities, axis=0)
    report = dict(
        view=view,
        signals=signals,
        mean_partition_accuracy=float(
            np.mean([accuracy_score(y, p >= THRESHOLD) for p in probabilities])
        ),
        averaged_oof_accuracy=float(accuracy_score(y, averaged >= THRESHOLD)),
        averaged_oof_log_loss=float(log_loss(y, averaged)),
    )

    model = LogisticRegression(C=1.0, max_iter=2000).fit(
        np.vstack(X), np.concatenate([y] * len(X))
    )
    p = model.predict_proba(np.column_stack(test_columns))[:, 1]
    submission = pd.DataFrame(
        dict(battle_id=test["battle_id"], player_won=(p >= THRESHOLD).astype(int))
    )
    path = output / f"submission_{name}.csv"
    submission.to_csv(path, index=False)
    report["coefficients"] = model.coef_.round(4).tolist()[0]
    report["submission"] = path.name
    print(json.dumps(report, indent=2), flush=True)
    return path
