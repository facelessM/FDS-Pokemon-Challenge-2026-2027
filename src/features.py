"""Construct each feature view using the outer training fold.

Training rows receive records and auxiliary estimates from inner held-out groups.
Validation and test rows use estimators fitted on the whole outer training fold, or,
with FEATURE_FIXES, the average of the inner estimators (see src/options.py).
"""

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .data import build_species_table, build_move_vocab, build_top_species
from .base_features import build_features
from .team_context import TeamContext
from .tactics import TacticalContext
from .move_records import MoveRecords
from .knockouts import make_events, model as auxiliary_model, aggregate
from . import state as deep
from . import coverage as tf
from .species import vocabulary, species_features, matrix
from .forecast import crossfit_forecast
from .pressure import damage_samples, fit_power, EmpiricalTactics
from .options import FEATURE_FIXES

BASE_GROUPS = [
    "starter",
    "state_t30",
    "sacrifice",
    "move_usage",
    "move_effects",
    "alive_species",
]


def subset(values, indices):
    return [values[int(i)] for i in indices]


def numeric(rows):
    frame = pd.DataFrame(rows).fillna(0).astype(np.float32)
    assert np.isfinite(frame.to_numpy()).all()
    return frame


def encodings(train_views, y, groups, target_views):
    """Build posterior team records without a battle's own outcome."""
    parts = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
        np.zeros(len(y)), y, groups
    ):
        assert not set(groups[a]) & set(groups[b])
        model = TeamContext(enriched=True, moves=True).fit(subset(train_views, a), y[a])
        part = model.transform(subset(train_views, b), "posterior")
        part.index = b
        parts.append(part)
    train = pd.concat(parts).sort_index().astype(np.float32)
    if FEATURE_FIXES:
        outs = []
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
            np.zeros(len(y)), y, groups
        ):
            model = TeamContext(enriched=True, moves=True).fit(subset(train_views, a), y[a])
            outs.append(model.transform(target_views, "posterior").astype(np.float32))
        valid = sum(outs) / len(outs)
    else:
        model = TeamContext(enriched=True, moves=True).fit(train_views, y)
        valid = model.transform(target_views, "posterior").astype(np.float32)
    return {"post": train}, {"post": valid}


def fold_features(battles, views, y, groups, tr, va, chart, static):
    bt = subset(battles, tr)
    vt = subset(views, tr)
    vv = subset(views, va)
    context = {
        "species_table": build_species_table(bt),
        "move_vocab": build_move_vocab(bt),
        "top_species": build_top_species(bt),
    }
    # No outcomes occur in these numeric features or in TacticalContext's outputs.
    base = (
        build_features(battles, BASE_GROUPS, context)
        .drop(columns="battle_id")
        .astype(np.float32)
    )
    left = {k: v.iloc[tr].reset_index(drop=True) for k, v in static.items()}
    right = {k: v.iloc[va].reset_index(drop=True) for k, v in static.items()}
    left["base"] = base.iloc[tr].reset_index(drop=True)
    right["base"] = base.iloc[va].reset_index(drop=True)
    e1, e2 = encodings(vt, y[tr], groups[tr], vv)
    left.update(e1)
    right.update(e2)
    tactical = TacticalContext().fit(bt, vt, chart, context["top_species"])
    for name, func in [("threat", tactical.threat_features)]:
        feat = numeric([func(b, v) for b, v in zip(battles, views)])
        left[name] = feat.iloc[tr].reset_index(drop=True)
        right[name] = feat.iloc[va].reset_index(drop=True)
    return dict(train=left, valid=right, tr=tr, va=va, ytr=y[tr], yva=y[va])


def calibrate_p2(p, M, source, tr):
    """P2 knockouts are mostly unrecorded, so the model learns P1 knockouts only.
    Shift P2 scores on the logit scale until the training fold expects as many
    P2 knockouts per battle as P1 has recorded ones."""
    in_tr = M.battle.isin(tr).to_numpy()
    target_rate = M.loc[source, "target"].sum() / len(tr)
    p2 = (M.side == "p2").to_numpy()
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit[p2 & in_tr] + mid)))).sum() / len(tr)
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return np.where(p2, 1 / (1 + np.exp(-(logit + (lo + hi) / 2))), p)


def matrices(context, tr, va):
    """Combine recorded state, team records, recovery and uncertain departures."""
    B, V, y, groups, chart = (
        context[key] for key in ["battles", "views", "y", "groups", "chart"]
    )
    assert not set(groups[tr]) & set(groups[va])
    base = fold_features(B, V, y, groups, tr, va, chart, context["static"])
    stats = deep.stats_context([B[i] for i in tr])
    extra = {"resources": context["resources"]}
    for name, fn in [("exits", deep.departures), ("switches", deep.switches)]:
        extra[name] = numeric([fn(b, v, stats, chart) for b, v in zip(B, V)])
    X, M = make_events(B, V, stats, chart, include_entry=True)
    source = M.battle.isin(tr) & (M.side == "p1")
    for c in ["victim", "attacker", "attack_move", "previous_own_move"]:
        X[c] = pd.Categorical(X[c], categories=sorted(X.loc[source, c].unique()))
    p = np.zeros(len(M))
    for a, b in StratifiedGroupKFold(2, shuffle=True, random_state=347).split(
        np.zeros(len(tr)), y[tr], groups[tr]
    ):
        assert not set(groups[tr[a]]) & set(groups[tr[b]])
        fit = M.battle.isin(tr[a]) & (M.side == "p1")
        query = M.battle.isin(tr[b])
        est = auxiliary_model().fit(X.loc[fit], M.loc[fit, "target"])
        p[query] = est.predict_proba(X.loc[query])[:, 1]
    query = M.battle.isin(va)
    if FEATURE_FIXES:
        halves = []
        for a, b in StratifiedGroupKFold(2, shuffle=True, random_state=347).split(
            np.zeros(len(tr)), y[tr], groups[tr]
        ):
            fit = M.battle.isin(tr[a]) & (M.side == "p1")
            halves.append(auxiliary_model().fit(X.loc[fit], M.loc[fit, "target"]))
        p[query] = np.mean([e.predict_proba(X.loc[query])[:, 1] for e in halves], axis=0)
        p = calibrate_p2(p, M, source, tr)
    else:
        est = auxiliary_model().fit(X.loc[source], M.loc[source, "target"])
        p[query] = est.predict_proba(X.loc[query])[:, 1]
    # Knockout risk per battle and P2 species, used by the tactical view.
    riskmap = [dict() for _ in range(len(B))]
    for index, row in M[M.side == "p2"].iterrows():
        riskmap[int(row.battle)][row.species] = float(p[index])
    context["riskmap"] = riskmap
    top = [
        name
        for name, count in Counter(
            name for i in tr for name in V[i]["team"]
        ).most_common(12)
    ]
    extra["entryko"] = aggregate(V, M, p, top)
    parts = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
        np.zeros(len(tr)), y[tr], groups[tr]
    ):
        assert not set(groups[tr[a]]) & set(groups[tr[b]])
        est = MoveRecords().fit([V[i] for i in tr[a]], y[tr[a]])
        z = est.transform([V[i] for i in tr[b]])
        z.index = tr[b]
        parts.append(z)
    if FEATURE_FIXES:
        outs = []
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
            np.zeros(len(tr)), y[tr], groups[tr]
        ):
            est = MoveRecords().fit([V[i] for i in tr[a]], y[tr[a]])
            outs.append(est.transform([V[i] for i in va]))
        z = sum(outs) / len(outs)
    else:
        est = MoveRecords().fit([V[i] for i in tr], y[tr])
        z = est.transform([V[i] for i in va])
    z.index = va
    parts.append(z)
    extra["sets"] = pd.concat(parts).sort_index()
    old = ["base", "post", "obs", "threat"]
    new = ["exits", "resources", "switches", "entryko", "sets"]
    A = pd.concat(
        [base["train"][key] for key in old]
        + [extra[key].iloc[tr].reset_index(drop=True) for key in new],
        axis=1,
    )
    Z = pd.concat(
        [base["valid"][key] for key in old]
        + [extra[key].iloc[va].reset_index(drop=True) for key in new],
        axis=1,
    )
    assert A.columns.is_unique and A.columns.equals(Z.columns)
    assert np.isfinite(A.to_numpy()).all() and np.isfinite(Z.to_numpy()).all()
    assert not {"battle_id", "player_won"} & set(A.columns)
    return A, Z


def tactical_joint(ctx, table, b, v, risk):
    scenarios, _, _ = ctx.rosters(v)
    m = tf.BattleMatrices(ctx, table, b, v, scenarios, modes=["consistent"])
    parts = {"consistent": tf.threat_features(m, scenarios, "consistent")}
    parts["identity"] = tf.identity_features(ctx, b, v)
    parts["tempo"] = tf.active_features(m)
    parts.update(tf.strategy_features(m, scenarios, tf.hazards(b, ctx), risk))
    old_b, old_v = tf.prefix_view(b, v)
    old_m = tf.BattleMatrices(
        ctx, table, old_b, old_v, scenarios, modes=["consistent"], worlds=False
    )
    old = tf.strategy_features(
        old_m, scenarios, tf.hazards(old_b, ctx), {}, coverage_only=True
    )["coverage"]
    parts["progress"] = {
        k.replace("coverage_", "progress_"): value - old[k]
        for k, value in parts["coverage"].items()
        if "_risk_" not in k
    }
    return parts


def additional(context, prefixes, tr, va, A, Z):
    """Fit pressure, HP forecasts and coverage evidence within the outer fold."""
    B, V, y, groups, chart = (
        context[k] for k in ["battles", "views", "y", "groups", "chart"]
    )
    fitB = [B[i] for i in tr]
    fitV = [V[i] for i in tr]
    top = build_top_species(fitB)
    ctx = TacticalContext().fit(fitB, fitV, chart, top)
    samples = damage_samples(B, ctx)
    which = np.full(len(B), -1, int)
    power = []
    for a, b in StratifiedGroupKFold(2, shuffle=True, random_state=347).split(
        np.zeros(len(tr)), y[tr], groups[tr]
    ):
        assert not set(groups[tr[a]]) & set(groups[tr[b]])
        power.append(EmpiricalTactics.from_context(ctx, fit_power(samples, tr[a])))
        which[tr[b]] = len(power) - 1
    if FEATURE_FIXES:

        def pressure_row(i, b, v):
            # Rows outside the training fold average the two half models.
            if which[i] >= 0:
                return power[which[i]].threat_features(b, v)
            parts = [h.threat_features(b, v) for h in power]
            return {
                c: (parts[0].get(c, 0.0) + parts[1].get(c, 0.0)) / 2
                for c in set(parts[0]) | set(parts[1])
            }

        pressure = numeric([pressure_row(i, b, v) for i, (b, v) in enumerate(zip(B, V))])
    else:
        power.append(EmpiricalTactics.from_context(ctx, fit_power(samples, tr)))
        which[va] = len(power) - 1
        pressure = numeric(
            [power[which[i]].threat_features(b, v) for i, (b, v) in enumerate(zip(B, V))]
        )
    top_species, _ = vocabulary(fitV)
    species = matrix([species_features(v, top_species) for v in V])
    forecast, _ = crossfit_forecast(prefixes, y, groups, tr, va)
    del power, samples, ctx
    tc = tf.Context().fit(fitB, fitV, chart, top)
    rows, _ = tf.evidence(fitB, fitV, tc)
    tables = [tf.ResponseTable(rows)]
    which = np.zeros(len(B), int)
    for a, b in StratifiedGroupKFold(2, shuffle=True, random_state=41071).split(
        np.zeros(len(tr)), y[tr], groups[tr]
    ):
        allowed = set(groups[tr[a]])
        assert not allowed & set(groups[tr[b]])
        tables.append(tf.ResponseTable([r for r in rows if r[0] in allowed]))
        which[tr[b]] = len(tables) - 1
    riskcols = [c for c in A if c.startswith("ko_p2_risk_")]
    risks = np.zeros((len(B), len(riskcols)), np.float32)
    risks[tr] = A[riskcols].to_numpy()
    risks[va] = Z[riskcols].to_numpy()
    output = {}
    training = set(int(i) for i in tr)
    for i, (b, v) in enumerate(zip(B, V)):
        if FEATURE_FIXES:
            # Every P2 species with departure evidence, not only the common ones.
            risk = dict(context["riskmap"][i])
        else:
            risk = {c[len("ko_p2_risk_") :]: float(q) for c, q in zip(riskcols, risks[i])}
        if FEATURE_FIXES and i not in training:
            halves = [tactical_joint(tc, tables[k], b, v, risk) for k in (1, 2)]
            parts = {
                name: {
                    c: (halves[0][name][c] + halves[1][name].get(c, 0.0)) / 2
                    for c in halves[0][name]
                }
                for name in halves[0]
            }
        else:
            parts = tactical_joint(tc, tables[which[i]], b, v, risk)
        for name, f in parts.items():
            output.setdefault(name, []).append(f)
    tactical = {n: numeric(rows) for n, rows in output.items()}
    return dict(
        pressure=pressure, species=species, forecast=forecast, tactical=tactical
    )


def components(A, Z, tr, va, new):
    """Keep four separate views so the blend can combine their different errors."""
    result = {"baseline": (A, Z)}
    result["species_forecast"] = tuple(
        pd.concat(
            [base]
            + [
                new[n].iloc[idx].reset_index(drop=True) for n in ["species", "forecast"]
            ],
            axis=1,
        )
        for base, idx in [(A, tr), (Z, va)]
    )
    p = []
    t = []
    for base, idx in [(A, tr), (Z, va)]:
        C = new["pressure"].iloc[idx].reset_index(drop=True)
        XX = base.copy()
        XX.loc[:, C.columns] = C.to_numpy()
        p.append(XX)
        C = new["tactical"]["consistent"].iloc[idx].reset_index(drop=True)
        XX = base.copy()
        XX.loc[:, C.columns] = C.to_numpy()
        t.append(
            pd.concat(
                [XX]
                + [
                    new["tactical"][n].iloc[idx].reset_index(drop=True)
                    for n in [
                        "identity",
                        "tempo",
                        "coverage",
                        "trades",
                        "hazard",
                        "progress",
                    ]
                ],
                axis=1,
            )
        )
    result["zero_power"] = tuple(p)
    result["tactical"] = tuple(t)
    for a, z in result.values():
        assert a.columns.is_unique and a.columns.equals(z.columns)
        assert not {"battle_id", "player_won"} & set(a.columns)
        assert np.isfinite(a.to_numpy()).all() and np.isfinite(z.to_numpy()).all()
    return result
