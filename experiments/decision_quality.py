"""Decision-quality features: did each player pick sensible moves at the time?

Hypothesis: the existing views describe the turn-30 position and a few switch
outcomes, but not whether a player's move choices were sound. Players who
repeatedly hit into immunities or resistances, ignore a revealed
super-effective option, or leave a Pokemon in against a super-effective threat
may be weaker and lose more of the remaining battle. Adding these rates to the
four core views should fix some battles that look even at turn 30.

A move at turn t hits the opponent active at turn t (switches happen first). We
score a *choice* only when neither side switched between t-1 and t, so the
player knew the target when choosing. A player's options are the damaging moves
that the same Pokemon revealed anywhere in the 30 visible turns. That is a lower
bound on its real moveset. Species types and base stats come from the training
battles, as in `src/state.py`. No turn after 30 and no outcome is used, so the
block needs no cross-fitting.

    caffeinate -i ./.venv/bin/python experiments/decision_quality.py --jobs 3
"""

import argparse
import json
import os
from pathlib import Path
import sys

SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
if __name__ == "__main__" and any(
    os.environ.get(name) != value for name, value in SETTINGS.items()
):
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **SETTINGS))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.evaluate import compare, evaluate, load_result, metrics
from src.data import load_type_chart
from src.observations import load
from src.state import pressure, stats_context, type_mult
from src.validation import COMPONENTS

CORE = [c for c in COMPONENTS if c != "raw_summary"]
FEATURES = ROOT / "outputs" / "store" / "experiments" / "decision_quality_features.pkl"


def damaging(m):
    return bool(m) and m["category"] != "status"


def side_features(b, side, opp, stats, chart):
    tl = b["battle_timeline"]
    # Every damaging move each Pokemon revealed in the visible prefix.
    known = {"own": defaultdict(dict), "opp": defaultdict(dict)}
    for t in tl:
        for who, s in [("own", side), ("opp", opp)]:
            m = t[s + "_move_details"]
            if damaging(m):
                known[who][t[s + "_pokemon_state"]["name"]][m["name"]] = m
    roster = [p["name"] for p in b["p1_team_details"]] if side == "p1" else sorted(
        {t["p2_pokemon_state"]["name"] for t in tl}
    )
    last = {}
    v = defaultdict(list)
    for i, t in enumerate(tl):
        me, them = t[side + "_pokemon_state"], t[opp + "_pokemon_state"]
        stayed = i > 0 and me["name"] == tl[i - 1][side + "_pokemon_state"]["name"]
        held = i > 0 and them["name"] == tl[i - 1][opp + "_pokemon_state"]["name"]
        a, d = stats.get(me["name"]), stats.get(them["name"])
        m = t[side + "_move_details"]
        if stayed and held and a and d and damaging(m) and me["status"] != "fnt":
            mult = type_mult(m["type"], d, chart)
            options = list(known["own"][me["name"]].values())
            pres = [pressure(o, a, d, me, them, chart) for o in options]
            mults = [type_mult(o["type"], d, chart) for o in options]
            chosen = pressure(m, a, d, me, them, chart)
            best = max(pres) if pres else chosen
            late = i >= 20
            v["immune"].append(mult == 0)
            v["resisted"].append(0 < mult < 1)
            v["super"].append(mult > 1)
            v["log_mult"].append(np.log2(max(mult, 0.25)))
            v["avoidable_immune"].append(mult == 0 and max(mults) > 0)
            v["missed_super"].append(mult <= 1 and max(mults) > 1)
            # Relative regret is only meaningful for moves with a power estimate.
            if m["base_power"] > 0 and best > 0:
                v["regret"].append((best - chosen) / best)
                if late:
                    v["late_regret"].append((best - chosen) / best)
            if late:
                v["late_bad"].append(mult < 1)
        # Stayed in while the opponent's revealed moves hit super-effectively.
        if stayed and held and a and d and me["status"] != "fnt":
            threats = known["opp"][them["name"]].values()
            incoming = max((type_mult(o["type"], a, chart) for o in threats), default=1.0)
            outgoing = max(
                (type_mult(o["type"], d, chart) for o in known["own"][me["name"]].values()),
                default=1.0,
            )
            threatened = incoming > 1 and outgoing <= 1
            v["stayed_threatened"].append(threatened)
            if threatened:
                bench = [
                    n
                    for n in roster
                    if n != me["name"]
                    and last.get(n, {"status": "nostatus"})["status"] != "fnt"
                    and n in stats
                ]
                answer = any(
                    max((type_mult(o["type"], stats[n], chart) for o in threats), default=1)
                    < 1
                    for n in bench
                )
                v["stayed_with_answer"].append(answer)
        last[me["name"]] = me
    out = {}
    pre = f"dq_{side}_"
    out[pre + "choices"] = len(v["immune"])
    for key in [
        "immune", "resisted", "super", "log_mult", "avoidable_immune", "missed_super",
        "regret", "late_regret", "late_bad", "stayed_threatened",
    ]:
        vals = np.asarray(v[key], float)
        out[pre + key + "_rate"] = vals.mean() if len(vals) else np.nan
    for key in ["avoidable_immune", "missed_super", "stayed_with_answer"]:
        out[pre + key + "_count"] = float(np.sum(v[key]))
    return out


def battle_features(b, stats, chart):
    row = {}
    row.update(side_features(b, "p1", "p2", stats, chart))
    row.update(side_features(b, "p2", "p1", stats, chart))
    for k in list(row):
        if k.startswith("dq_p1_"):
            row["dq_diff_" + k[6:]] = row[k] - row["dq_p2_" + k[6:]]
    return row


def build(data):
    train = load(data / "train.jsonl")
    stats = stats_context(train)
    chart = load_type_chart(data)
    frame = pd.DataFrame([battle_features(b, stats, chart) for b in train])
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(FEATURES)
    return frame, np.array([int(b["player_won"]) for b in train])


def add_block(fold_data, names):
    block = pd.read_pickle(FEATURES)
    views = dict(fold_data["views"])
    for name in names:
        X, Q = views[name]
        views[name] = (
            pd.concat([X, block.iloc[fold_data["tr"]].reset_index(drop=True)], axis=1),
            pd.concat([Q, block.iloc[fold_data["va"]].reset_index(drop=True)], axis=1),
        )
    return dict(fold_data, views=views)


def add_to_core(fold_data):
    return add_block(fold_data, CORE)


def univariate(frame, y):
    """Quick signal check before fitting anything: rank correlation with the win."""
    rows = []
    for c in frame:
        ok = frame[c].notna()
        if ok.sum() > 100 and frame.loc[ok, c].nunique() > 1:
            r = pd.Series(frame.loc[ok, c]).rank().corr(pd.Series(y[ok]).rank())
            rows.append((c, float(ok.mean()), float(frame.loc[ok, c].mean()), r))
    out = pd.DataFrame(rows, columns=["feature", "coverage", "mean", "spearman_with_p1_win"])
    return out.sort_values("spearman_with_p1_win", key=abs, ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--store", type=Path, default=ROOT / "outputs" / "store")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    frame, y = build(args.data)
    print(f"{frame.shape[1]} decision-quality columns for {len(frame)} battles")
    print(univariate(frame, y).head(20).to_string(index=False), flush=True)
    if args.features_only:
        return
    result = evaluate(
        args.store, name="decision_quality", components=CORE,
        transform=add_to_core, jobs=args.jobs,
    )
    reference = load_result(args.store, "reference")
    # Swap the four core components and keep the reference raw view.
    raw = {k: reference[k] for k in ["y", "test_ids"]}
    keep = list(reference["names"]).index("raw_summary")
    combined = dict(
        names=np.asarray(CORE + ["raw_summary"]),
        cv=np.concatenate([result["cv"], reference["cv"][:, :, [keep]]], axis=2),
        test=np.concatenate([result["test"], reference["test"][:, :, [keep]]], axis=2),
        **raw,
    )
    print(json.dumps(dict(new=metrics(combined), reference=metrics(reference)), indent=2))
    print("Five-view blend, decision quality vs reference:")
    summary = {"blend": compare(combined, reference, args.store)}
    for name in CORE:
        only = {name: 1.0}
        print(f"{name} alone:")
        summary[name] = compare(combined, reference, args.store, only, only)
    (args.store / "experiments" / "decision_quality.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
