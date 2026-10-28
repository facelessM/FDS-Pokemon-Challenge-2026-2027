# A wild Gengar Appears!

**Pokémon battle outcome prediction** · Foundations of Data Science, Sapienza Università di Roma, 2026/2027

We predict whether Player 1 wins from its team, Player 2's lead and the first
30 turns of a Gen 2 OU battle. Our selected model averages five feature views
across three grouped CV partitions. Averaging the three out-of-fold probabilities
before classification gives **0.7103 accuracy** on the 10,000 training battles.
The earlier state model scores **0.7026** under the same averaging rule.

These battles were also used to compare candidates. The CV result is a development
estimate, not a new independent holdout score.

## Team

| Name | Student ID | Email |
|---|---|---|
| Özgür Cihan Karahan | 2233702 | karahan.2233702@studenti.uniroma1.it |
| Gabriele Fiacconi | 2060444 | fiacconi.2060444@studenti.uniroma1.it |
| Musa Alasgarov | 2242081 | alasgarov.2242081@studenti.uniroma1.it |
| Baran Berkay Hökelek | 2243542 | hokelek.2243542@studenti.uniroma1.it |

## Development history

We keep the earlier submissions and unsuccessful experiments in the
[development history](docs/development_history.md), with the numerical record in
[history.json](results/history.json). The report also follows this progression.

| Stage | Main change | Recorded CV accuracy |
|---|---|---:|
| Starter | Team and lead averages | 0.5178 |
| E01 | Last-observed battle state | 0.6671 |
| E05 | Separate sacrifice moves | 0.6712 |
| Wide tree blend | Move usage, effects and surviving species | 0.6875 |
| Compact blend | Role counts and 176 selected columns | 0.6885 |
| Team records | Inner-fold records and deterministic roster matching | 0.6947 |
| First win conditions | Species-pair HP exchanges | 0.6939 |
| State reconstruction | Recovery, uncertain departures and move records | 0.7013 |
| Species and forecasts | Species HP/status and five-turn changes | 0.7036 |
| Tactical coverage | Move hypotheses and remaining defensive answers | 0.7051 |
| Matched five-view mixture | Three OOF probabilities averaged per battle | **0.7103** |

The first seven rows use the original seed-42 CV. The next three are means of
three grouped partitions. The final row scores averaged OOF probabilities, whose
separate-partition mean is 0.7060. These are historical milestones with different
evaluation procedures, not one directly comparable score series. The team-record
row is after its tie-breaking fix; its earlier submitted file used the previous version.

Trajectory and basic type summaries, hierarchy/CatBoost mixtures, uncertainty
features, ordered observations, state-signal trials and learned mixture weights
are also documented, including cases that did not justify a replacement.

## Method

Each fold combines five probabilities with equal weights:

1. Reconstructed battle state, recovery resources, uncertain P2 knockout evidence and team/move records.
2. The same state with species-specific HP and status, plus a forecast learned from visible five-turn HP changes.
3. The same state with consistent move-set hypotheses, defensive coverage, hazards and trade features.
4. The same state with learned pressure for attacks whose recorded base power is zero.
5. A separate LightGBM model of the recorded observations, including identities, last states and row summaries.

The first four views each average LightGBM and XGBoost. Vocabularies and learned
statistics come from the outer training fold. Outcome records and auxiliary
estimates for its training rows use inner grouped splits.

Each partition uses five folds, with seeds `20261026`, `20261105` and `193`.
Identical timelines stay together. Each battle therefore receives three
out-of-fold mixture probabilities, which we average before applying `0.5`.

| Evaluation | Earlier state model | Five-view model |
|---|---:|---:|
| Mean accuracy of the three separate partitions | 0.7013 | 0.7060 |
| Accuracy after averaging the three OOF probabilities | 0.7026 | **0.7103** |

For test prediction, each battle uses one fold ensemble from each partition.
A timeline already present in training uses its held-out fold. New timelines
use a fixed hash. This preserves the evaluated training size and number of
predictions per battle. We do not average all fifteen fold ensembles or refit
the winner classifiers on all 10,000 rows.

[Methodology](docs/methodology.md) explains the features, approximations, inner
splits and results. [Reference results](results/reference.json) record the
expected metrics and prediction hashes.

## Reproduce

Use Python 3.12 (tested with 3.12.14) and the versions in `requirements.txt`. Place the four competition
files in `data/`: `train.jsonl`, `test.jsonl`, `sample_submission.csv` and
`type_chart_gen2.csv`.

```bash
python -m pip install -r requirements.txt
python run.py predict --data data --jobs 2
```

This builds every feature from the competition files, trains all fifteen fold
ensembles, evaluates OOF predictions and writes
`outputs/predict/submission_matched_core_summary.csv`. The folder also contains
per-fold probabilities, logs and `metrics.json`. Two workers are the default;
use `--jobs 1` if memory is limited. This is a full training run, not a quick demo.

To run CV without loading the test set:

```bash
python run.py cv --data data --jobs 2
python -m unittest discover -s tests -v
```

Interrupted runs reuse completed folds only when the input files, source and
package versions match. Use a new `--output` directory after changing any of
them. To rebuild the CSV and metrics from completed folds:

```bash
python run.py collect --data data --output outputs/predict
```

## Code layout

- `run.py` is the only training entry point.
- `src/pipeline.py` prepares inputs, runs folds and writes predictions.
- `src/validation.py` holds splits, routing, weights and the threshold.
- `src/features.py` assembles the four core views and their inner estimates.
- The other `src/` modules contain the corresponding feature calculations.
- `notebooks/kaggle.ipynb` clones this repository and runs the same entry point.

Main contains the selected training method and its development history. Earlier experiment code and results remain
in [the preceding revision](https://github.com/facelessM/FDS-Pokemon-Challenge-2026-2027/tree/f4c233b8e9523779b3a29c8eec1e3d229323f158).

## Competition rules

We use classical machine learning and only the competition files and supplied
Gen 2 type chart. Species statistics, move examples and roster frequencies are
learned from the training battles. We use no external databases, replays,
usage statistics, pretrained models or manually collected tables. Test predictions
are not used to select features, weights, routing or the threshold.
