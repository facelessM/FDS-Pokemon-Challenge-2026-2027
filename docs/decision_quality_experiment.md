# Decision-quality features

Nov 5, 2026 · Baran Berkay Hökelek · Not adopted

## Question

The selected model describes the battle at turn 30: remaining HP, faints,
status, revealed moves, team records and tactical coverage. It also scores how
switches turned out (`switches()` in `src/state.py`: resisted or immune entries,
pressure saved, worse entries). It does not ask whether each player chose
sensible *moves*.

The hypothesis was that move-choice quality is a proxy for player skill. A
player who hits into immunities or resistances, ignores a revealed
super-effective move, or leaves a Pokémon in against a super-effective threat
may also play the unseen remainder of the battle worse. That could separate
battles that look even at turn 30, where most errors are.

## Which choices are scored

Each timeline entry records the state *after* both moves. Switches happen
first, so a move at turn t hits the opponent active at turn t. A player chose
that move while seeing the opponent active at turn t-1.

We therefore score a choice only when **neither side switched** between t-1 and
t and the move was damaging. Then the player knew the target. Moves into a
Pokémon that just switched in are left out. Those reflect the opponent's
prediction, which the existing switch features already describe.

A Pokémon's options are the damaging moves it revealed anywhere in the 30
turns. This is a lower bound on its real moveset. A player knows their own full
set, so using moves revealed later in the prefix is not hindsight for that
player.

## Features

`experiments/decision_quality.py` builds 42 columns: 13 per side and 16 P1 − P2
differences.

| Feature | Meaning |
|---|---|
| `choices` | Number of scored choices |
| `immune_rate`, `resisted_rate`, `super_rate` | Share of scored attacks with type multiplier 0, between 0 and 1, above 1 |
| `log_mult_rate` | Mean log2 type multiplier (0.25 floor) |
| `avoidable_immune_rate` / `_count` | Hit an immunity although another revealed move would have done damage |
| `missed_super_rate` / `_count` | Chose a neutral or worse move although a revealed move was super-effective |
| `regret_rate` | Mean (best − chosen) / best under the `pressure()` damage proxy, for moves with known base power |
| `late_regret_rate`, `late_bad_rate` | Regret and resisted-or-immune share in turns 20–30 |
| `stayed_threatened_rate` | Stayed in while the opponent's revealed moves hit super-effectively and ours did not |
| `stayed_with_answer_count` | Of those turns, how many had a living teammate that resisted every revealed threat |

The block uses no outcomes and no turn after 30, so it needs no cross-fitting.
Species types and base stats come from the training battles, as in the
pipeline.

## Procedure

- Feature build: about 20 seconds for all 10,000 training battles.
- The block was appended to the four core views (baseline, species forecast,
  tactical, zero power). The raw view was kept from the cached reference.
- Classifiers were refitted with the selected LightGBM + XGBoost settings on
  the cached features of all 15 outer folds (grouped partitions 20261026,
  20261105 and 193). This took about 15 minutes with three workers.
- The reference is the cached reproduction of the selected model, 0.7098. It
  differs from the recorded 0.7103 because one team-record feature
  occasionally evaluates to about 4e-20 instead of 0. Both runs read the same
  cached features, so the comparison is paired.

## Results

**How often the scored situations occur.** Each side makes about 14 damaging
moves in the 30 turns. About 6 come with both Pokémon staying in, and about 5
hit a Pokémon that switched in that turn. Immune hits are 1.3% of scored
choices. No column has a rank correlation above 0.07 with the outcome. Some
signs run against the hypothesis: P2 skipping a revealed super-effective move
goes slightly with P2 winning.

**Five-view blend, averaged out-of-fold.**

| | Cached reference | With decision quality |
|---|---:|---:|
| Averaged OOF accuracy | 0.7098 | 0.7103 |
| Partition accuracies | 0.7056, 0.7035, 0.7066 | 0.7065, 0.7054, 0.7058 |
| Log loss | 0.55285 | 0.55312 |

The gain is +0.0005: 82 battles fixed, 77 broken, bootstrap interval −0.0020
to +0.0030.

**Single views.**

| View | Reference | New | Change |
|---|---:|---:|---:|
| Baseline | 0.7028 | 0.7039 | +0.0011 |
| Species forecast | 0.7061 | 0.7047 | −0.0014 |
| Tactical | 0.7057 | 0.7071 | +0.0014 |
| Zero power | 0.7059 | 0.7031 | −0.0028 |

Log loss is worse in all four views.

## Interpretation

The gain is within run-to-run noise (about ±0.001), and log loss worsens, so
the block was **not adopted**. Three reasons are likely:

1. **Too few scored choices.** Gen 2 OU switches often, so only about six
   choices per side meet the clean condition.
2. **The obvious move is often not the best one.** Status moves, setup moves,
   scouting and switch prediction can make a weaker hit the right choice. Only
   avoidable immune hits are clear blunders, and they are rare.
3. **The turn-30 state already reflects past choices.** Bad decisions show up
   as lost HP and fainted Pokémon, which the model already sees.

A possible follow-up would aggregate the same rates per recurring team across
training battles, using inner folds as for the team records, so that habits
are measured over many battles instead of six choices. Given these results, a
large gain is not expected.

## Reproduce

Build the feature store once, then run the experiment:

```bash
./.venv/bin/python experiments/store.py --data data --output outputs/store --mode cv --jobs 2
./.venv/bin/python experiments/evaluate.py reproduce --store outputs/store --jobs 2
caffeinate -i ./.venv/bin/python experiments/decision_quality.py --jobs 3
```

The second command saves the `reference` run that the experiment compares
against. Use `--features-only` to build the columns and print their
correlations without refitting. Results are written to
`outputs/store/experiments/decision_quality.{npz,json}`.

The numbers are also recorded in Section 18 of
[development_history.md](development_history.md) and under
`decision_quality_trial` in [history.json](../results/history.json).
