"""Build the feature views for one outer fold.

Everything learned from data (vocabularies, records, auxiliary models) is fitted
on the outer training rows only. Training rows get their record and auxiliary
values from inner splits, so no row sees an estimate fitted on its own battle.
Validation and test rows get the estimate fitted on the whole training fold or,
with FEATURE_FIXES, the average of the inner estimates (see src/options.py).

Naming: `train_rows` and `query_rows` index the battles of the fold (query rows
are the validation battles followed by the routed test battles). `train_X` and
`query_X` are the matching feature tables.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import coverage
from . import state
from .base_features import build_features
from .data import build_move_vocab, build_species_table, build_top_species
from .forecast import crossfit_forecast
from .knockouts import aggregate, make_events
from .knockouts import model as knockout_model
from .move_records import MoveRecords
from .options import FEATURE_FIXES
from .pressure import EmpiricalTactics, damage_samples, fit_power
from .species import matrix, species_features, vocabulary
from .tactics import TacticalContext
from .team_context import TeamContext

BASE_GROUPS = [
    "starter",
    "state_t30",
    "sacrifice",
    "move_usage",
    "move_effects",
    "alive_species",
]
TACTICAL_GROUPS = ["identity", "tempo", "coverage", "trades", "hazard", "progress"]

# Seeds of the inner splits. Records use five folds, auxiliary models two.
RECORD_SEED = 99
KNOCKOUT_SEED = 347
RESPONSE_SEED = 41071


def subset(values, indices):
    return [values[int(i)] for i in indices]


def numeric(rows):
    """Feature dicts to a float32 table; a key missing in a row means 0."""
    frame = pd.DataFrame(rows).fillna(0).astype(np.float32)
    assert np.isfinite(frame.to_numpy()).all()
    return frame


def inner_splits(rows, y, groups, n_splits, seed):
    """Grouped stratified split of the training rows into (fit, held-out) parts."""
    split = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for a, b in split.split(np.zeros(len(rows)), y[rows], groups[rows]):
        assert not set(groups[rows[a]]) & set(groups[rows[b]])
        yield rows[a], rows[b]


def team_records(train_views, y, groups, query_views):
    """Team records (src/team_context.py) without a battle's own outcome."""
    rows = np.arange(len(y))
    parts, inner = [], []
    for fit, held in inner_splits(rows, y, groups, 5, RECORD_SEED):
        model = TeamContext(enriched=True, moves=True).fit(
            subset(train_views, fit), y[fit]
        )
        part = model.transform(subset(train_views, held), "posterior")
        part.index = held
        parts.append(part)
        inner.append(model)
    train = pd.concat(parts).sort_index().astype(np.float32)
    if FEATURE_FIXES:
        outs = [m.transform(query_views, "posterior").astype(np.float32) for m in inner]
        query = sum(outs) / len(outs)
    else:
        model = TeamContext(enriched=True, moves=True).fit(train_views, y)
        query = model.transform(query_views, "posterior").astype(np.float32)
    return train, query


def first_features(battles, views, y, groups, train_rows, query_rows, chart, static):
    """The first feature blocks: turn-30 state, moves, species and team records."""
    train_battles = subset(battles, train_rows)
    train_views = subset(views, train_rows)
    context = {
        "species_table": build_species_table(train_battles),
        "move_vocab": build_move_vocab(train_battles),
        "top_species": build_top_species(train_battles),
    }
    # These blocks use no outcomes, so they are computed for all rows at once.
    base = (
        build_features(battles, BASE_GROUPS, context)
        .drop(columns="battle_id")
        .astype(np.float32)
    )
    tactical = TacticalContext().fit(
        train_battles, train_views, chart, context["top_species"]
    )
    threat = numeric([tactical.threat_features(b, v) for b, v in zip(battles, views)])
    blocks = {"base": base, **static, "threat": threat}

    def rows(frame, index):
        return frame.iloc[index].reset_index(drop=True)

    train = {name: rows(frame, train_rows) for name, frame in blocks.items()}
    query = {name: rows(frame, query_rows) for name, frame in blocks.items()}
    train["post"], query["post"] = team_records(
        train_views, y[train_rows], groups[train_rows], subset(views, query_rows)
    )
    return train, query


def calibrate_p2(p, events, source, train_rows):
    """Shift P2 knockout scores so their total matches the recorded P1 rate.

    P2 knockouts are mostly unrecorded, so the model learns from P1 knockouts
    only. We shift P2 scores on the logit scale until the training fold expects
    as many P2 knockouts per battle as P1 has recorded ones.
    """
    in_train = events.battle.isin(train_rows).to_numpy()
    target_rate = events.loc[source, "target"].sum() / len(train_rows)
    p2 = (events.side == "p2").to_numpy()
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lo, hi = -10.0, 10.0
    for _ in range(60):  # bisection on the shift
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit[p2 & in_train] + mid)))).sum() / len(train_rows)
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return np.where(p2, 1 / (1 + np.exp(-(logit + (lo + hi) / 2))), p)


def knockout_scores(events, X, y, groups, train_rows, query_rows):
    """Score every departure event with a model trained on recorded P1 knockouts."""
    source = events.battle.isin(train_rows) & (events.side == "p1")
    for c in ["victim", "attacker", "attack_move", "previous_own_move"]:
        X[c] = pd.Categorical(X[c], categories=sorted(X.loc[source, c].unique()))
    p = np.zeros(len(events))
    halves = []
    for fit_rows, held_rows in inner_splits(train_rows, y, groups, 2, KNOCKOUT_SEED):
        fit = events.battle.isin(fit_rows) & (events.side == "p1")
        held = events.battle.isin(held_rows)
        model = knockout_model().fit(X.loc[fit], events.loc[fit, "target"])
        p[held] = model.predict_proba(X.loc[held])[:, 1]
        halves.append(model)
    query = events.battle.isin(query_rows)
    if FEATURE_FIXES:
        p[query] = np.mean(
            [m.predict_proba(X.loc[query])[:, 1] for m in halves], axis=0
        )
        p = calibrate_p2(p, events, source, train_rows)
    else:
        model = knockout_model().fit(X.loc[source], events.loc[source, "target"])
        p[query] = model.predict_proba(X.loc[query])[:, 1]
    return p


def move_records(views, y, groups, train_rows, query_rows):
    """Species and move-pair records (src/move_records.py), cross-fitted."""
    parts, inner = [], []
    for fit, held in inner_splits(train_rows, y, groups, 5, RECORD_SEED):
        model = MoveRecords().fit(subset(views, fit), y[fit])
        part = model.transform(subset(views, held))
        part.index = held
        parts.append(part)
        inner.append(model)
    if FEATURE_FIXES:
        outs = [m.transform(subset(views, query_rows)) for m in inner]
        query = sum(outs) / len(outs)
    else:
        model = MoveRecords().fit(subset(views, train_rows), y[train_rows])
        query = model.transform(subset(views, query_rows))
    query.index = query_rows
    parts.append(query)
    return pd.concat(parts).sort_index()


def matrices(context, train_rows, query_rows):
    """The state view: recorded state, team records, recovery and departures.

    Also stores in context["riskmap"] the knockout score of every P2 species in
    every battle, which the tactical view uses.
    """
    battles, views, y, groups, chart = (
        context[key] for key in ["battles", "views", "y", "groups", "chart"]
    )
    assert not set(groups[train_rows]) & set(groups[query_rows])
    train, query = first_features(
        battles, views, y, groups, train_rows, query_rows, chart, context["static"]
    )
    stats = state.stats_context(subset(battles, train_rows))
    extra = {"resources": context["resources"]}
    for name, fn in [("exits", state.departures), ("switches", state.switches)]:
        extra[name] = numeric([fn(b, v, stats, chart) for b, v in zip(battles, views)])

    X, events = make_events(battles, views, stats, chart, include_entry=True)
    p = knockout_scores(events, X, y, groups, train_rows, query_rows)
    riskmap = [dict() for _ in range(len(battles))]
    for index, row in events[events.side == "p2"].iterrows():
        riskmap[int(row.battle)][row.species] = float(p[index])
    context["riskmap"] = riskmap
    common = Counter(name for i in train_rows for name in views[i]["team"])
    top = [name for name, _ in common.most_common(12)]
    extra["entryko"] = aggregate(views, events, p, top)
    extra["sets"] = move_records(views, y, groups, train_rows, query_rows)

    first = ["base", "post", "obs", "threat"]
    later = ["exits", "resources", "switches", "entryko", "sets"]

    def table(blocks, rows):
        return pd.concat(
            [blocks[key] for key in first]
            + [extra[key].iloc[rows].reset_index(drop=True) for key in later],
            axis=1,
        )

    train_X, query_X = table(train, train_rows), table(query, query_rows)
    assert train_X.columns.is_unique and train_X.columns.equals(query_X.columns)
    assert (
        np.isfinite(train_X.to_numpy()).all() and np.isfinite(query_X.to_numpy()).all()
    )
    assert not {"battle_id", "player_won"} & set(train_X.columns)
    return train_X, query_X


def tactical_joint(ctx, table, battle, view, risk):
    """Coverage, trade, hazard and tempo features of one battle for the tactical view.

    "progress" is the change in coverage between turn 20 and turn 30, with the
    move information of turn 30 held fixed.
    """
    scenarios, _, _ = ctx.rosters(view)
    m = coverage.BattleMatrices(ctx, table, battle, view, scenarios)
    parts = {"consistent": coverage.threat_features(m, scenarios)}
    parts["identity"] = coverage.identity_features(ctx, battle, view)
    parts["tempo"] = coverage.active_features(m)
    parts.update(
        coverage.strategy_features(m, scenarios, coverage.hazards(battle, ctx), risk)
    )
    old_battle, old_view = coverage.prefix_view(battle, view)
    old_m = coverage.BattleMatrices(
        ctx, table, old_battle, old_view, scenarios, worlds=False
    )
    old = coverage.strategy_features(
        old_m, scenarios, coverage.hazards(old_battle, ctx), {}, coverage_only=True
    )["coverage"]
    parts["progress"] = {
        k.replace("coverage_", "progress_"): value - old[k]
        for k, value in parts["coverage"].items()
        if "_risk_" not in k
    }
    return parts


def zero_power_pressure(battles, views, y, groups, train_rows, ctx):
    """Threat features with learned pressure for zero-power attacks (src/pressure.py)."""
    samples = damage_samples(battles, ctx)
    which = np.full(len(battles), -1, int)  # which fitted model serves each row
    power = []
    for fit, held in inner_splits(train_rows, y, groups, 2, KNOCKOUT_SEED):
        power.append(EmpiricalTactics.from_context(ctx, fit_power(samples, fit)))
        which[held] = len(power) - 1
    if FEATURE_FIXES:

        def row(i, b, v):
            if which[i] >= 0:
                return power[which[i]].threat_features(b, v)
            # Rows outside the training fold average the two half models.
            parts = [h.threat_features(b, v) for h in power]
            return {
                c: (parts[0].get(c, 0.0) + parts[1].get(c, 0.0)) / 2
                for c in set(parts[0]) | set(parts[1])
            }

    else:
        power.append(EmpiricalTactics.from_context(ctx, fit_power(samples, train_rows)))
        which[which < 0] = len(power) - 1

        def row(i, b, v):
            return power[which[i]].threat_features(b, v)

    return numeric([row(i, b, v) for i, (b, v) in enumerate(zip(battles, views))])


def tactical_features(context, train_rows, query_rows, train_X, query_X):
    """The tactical view's extra blocks, one dict of tables per block name."""
    battles, views, y, groups, chart = (
        context[k] for k in ["battles", "views", "y", "groups", "chart"]
    )
    fit_battles, fit_views = subset(battles, train_rows), subset(views, train_rows)
    ctx = coverage.Context().fit(
        fit_battles, fit_views, chart, build_top_species(fit_battles)
    )
    # HP-response tables: one on the whole training fold, one per inner half.
    rows = coverage.evidence(fit_battles, fit_views, ctx)
    tables = [coverage.ResponseTable(rows)]
    which = np.zeros(len(battles), int)
    for fit, held in inner_splits(train_rows, y, groups, 2, RESPONSE_SEED):
        allowed = set(groups[fit])
        tables.append(coverage.ResponseTable([r for r in rows if r[0] in allowed]))
        which[held] = len(tables) - 1
    riskcols = [c for c in train_X if c.startswith("ko_p2_risk_")]
    risks = np.zeros((len(battles), len(riskcols)), np.float32)
    risks[train_rows] = train_X[riskcols].to_numpy()
    risks[query_rows] = query_X[riskcols].to_numpy()
    training = set(int(i) for i in train_rows)
    output = {}
    for i, (b, v) in enumerate(zip(battles, views)):
        if FEATURE_FIXES:
            # Every P2 species with departure evidence, not only the common ones.
            risk = dict(context["riskmap"][i])
        else:
            risk = {
                c[len("ko_p2_risk_") :]: float(q) for c, q in zip(riskcols, risks[i])
            }
        if FEATURE_FIXES and i not in training:
            halves = [tactical_joint(ctx, tables[k], b, v, risk) for k in (1, 2)]
            parts = {
                name: {
                    c: (halves[0][name][c] + halves[1][name].get(c, 0.0)) / 2
                    for c in halves[0][name]
                }
                for name in halves[0]
            }
        else:
            parts = tactical_joint(ctx, tables[which[i]], b, v, risk)
        for name, f in parts.items():
            output.setdefault(name, []).append(f)
    return {name: numeric(rows) for name, rows in output.items()}


def additional(context, prefixes, train_rows, query_rows, train_X, query_X):
    """Fit the blocks of the other three views inside the outer fold."""
    battles, views, y, groups, chart = (
        context[k] for k in ["battles", "views", "y", "groups", "chart"]
    )
    fit_battles, fit_views = subset(battles, train_rows), subset(views, train_rows)
    ctx = TacticalContext().fit(
        fit_battles, fit_views, chart, build_top_species(fit_battles)
    )
    pressure = zero_power_pressure(battles, views, y, groups, train_rows, ctx)
    top_species, _ = vocabulary(fit_views)
    species = matrix([species_features(v, top_species) for v in views])
    forecast = crossfit_forecast(prefixes, y, groups, train_rows, query_rows)
    del ctx
    tactical = tactical_features(context, train_rows, query_rows, train_X, query_X)
    return dict(
        pressure=pressure, species=species, forecast=forecast, tactical=tactical
    )


def components(train_X, query_X, train_rows, query_rows, new):
    """The four core views, kept separate so the blend can combine their errors.

    baseline: the state view. species_forecast: plus species HP and forecasts.
    zero_power: threat columns replaced by learned pressure. tactical: threat
    columns replaced by the consistent move-set version, plus coverage blocks.
    """

    def rows(frame, index):
        return frame.iloc[index].reset_index(drop=True)

    def replaced(base, block):
        out = base.copy()
        out.loc[:, block.columns] = block.to_numpy()
        return out

    result = {"baseline": (train_X, query_X)}
    pairs = [(train_X, train_rows), (query_X, query_rows)]
    result["species_forecast"] = tuple(
        pd.concat([base, rows(new["species"], i), rows(new["forecast"], i)], axis=1)
        for base, i in pairs
    )
    result["zero_power"] = tuple(
        replaced(base, rows(new["pressure"], i)) for base, i in pairs
    )
    result["tactical"] = tuple(
        pd.concat(
            [replaced(base, rows(new["tactical"]["consistent"], i))]
            + [rows(new["tactical"][n], i) for n in TACTICAL_GROUPS],
            axis=1,
        )
        for base, i in pairs
    )
    for a, z in result.values():
        assert a.columns.is_unique and a.columns.equals(z.columns)
        assert not {"battle_id", "player_won"} & set(a.columns)
        assert np.isfinite(a.to_numpy()).all() and np.isfinite(z.to_numpy()).all()
    return result
