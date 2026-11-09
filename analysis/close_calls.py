"""Where the selected models are right and wrong: accuracy by confidence.

Groups battles by how far the out-of-fold probability is from 0.5 and prints
the share of battles and the accuracy in each group, for the three selected
submissions. It reads finished runs, so run these first:

    python run.py predict --output outputs/original
    python run.py predict --feature-fixes --output outputs/fixed

Usage:
    python analysis/close_calls.py --original outputs/original --fixed outputs/fixed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.observations import load
from src.pipeline import read_pickle
from src.second_stage import (
    SimilarTeams,
    logit,
    similar_team_features,
    spikes_feature,
)

BINS = [0, 0.1, 0.2, 0.3, 0.5]
LABELS = ["< 10 points", "10-20", "20-30", "> 30"]


def five_view_oof(output):
    """Pooled OOF probability of the equal five-view blend."""
    oof = np.load(output / "oof.npz")
    return np.mean([c.mean(1) for c in oof["components"]], axis=0)


def stage_oof(data, output, view, signals):
    """Pooled OOF probability of the second stage, as `run.py stage` reports it."""
    inputs = read_pickle(output / "inputs.pkl.gz")
    oof = np.load(output / "oof.npz")
    y = inputs["y"]
    train, test = load(data / "train.jsonl"), load(data / "test.jsonl")
    if view == "blend":
        base = [c.mean(1) for c in oof["components"]]
    else:
        base = [c[:, 2] for c in oof["components"]]
    team, _ = similar_team_features(
        SimilarTeams(train, test), inputs, np.arange(len(y), len(y) + len(test))
    )
    columns = [[logit(b), t] for b, t in zip(base, team)]
    if "spikes" in signals:
        spikes = spikes_feature(train)
        columns = [c + [spikes] for c in columns]
    probabilities = []
    for x in [np.column_stack(c) for c in columns]:
        p = np.zeros(len(y))
        split = StratifiedGroupKFold(5, shuffle=True, random_state=7)
        for a, b in split.split(x, y, inputs["groups"]):
            model = LogisticRegression(C=1.0, max_iter=2000).fit(x[a], y[a])
            p[b] = model.predict_proba(x[b])[:, 1]
        probabilities.append(p)
    return np.mean(probabilities, axis=0)


def by_confidence(name, p, y):
    """Share and accuracy of battles by distance of p from 0.5."""
    d = pd.DataFrame({"right": (p >= 0.5) == y})
    d["distance"] = pd.cut(np.abs(p - 0.5), BINS, include_lowest=True, labels=LABELS)
    g = d.groupby("distance", observed=True)
    table = pd.DataFrame(
        {"share": (g.size() / len(d)).round(3), "accuracy": g.right.mean().round(3)}
    )
    print(f"\n{name}: pooled OOF accuracy {accuracy_score(y, p >= 0.5):.4f}")
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--original", type=Path, default=ROOT / "outputs/original")
    parser.add_argument("--fixed", type=Path, default=ROOT / "outputs/fixed")
    args = parser.parse_args()
    y = read_pickle(args.fixed / "inputs.pkl.gz")["y"]
    by_confidence("matched_core_summary", five_view_oof(args.original), y)
    by_confidence(
        "team_similarity", stage_oof(args.data, args.fixed, "blend", "team"), y
    )
    by_confidence(
        "tactical_team_spikes",
        stage_oof(args.data, args.fixed, "tactical", "team,spikes"),
        y,
    )


if __name__ == "__main__":
    main()
