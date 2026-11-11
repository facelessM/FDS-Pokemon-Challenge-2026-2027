# A wild Gengar Appears!

**Pokémon battle outcome prediction** · Foundations of Data Science, Sapienza Università di Roma, 2026/2027

We predict whether Player 1 wins from its team, Player 2's lead and the first
30 turns of a Gen 2 OU battle. Our models average five feature views across
three grouped CV partitions, and two of them add a small logistic regression
for similar-team records and Spikes.

| Selected submission | Averaged OOF accuracy | Public score | Private score |
|---|---:|---:|---:|
| `submission_matched_core_summary.csv`: five views | 0.7103 | 0.70466 | 0.69371 |
| `submission_team_similarity.csv`: corrected five views + similar teams | 0.7118 | 0.70200 | 0.69514 |
| `submission_tactical_team_spikes.csv`: corrected tactical view + similar teams and Spikes | 0.7085 | 0.70000 | **0.70200** |

We finished second on the private leaderboard with 0.70200. The single
tactical view scored best on private data, while the five-view models ended
between 0.6926 and 0.6951. These battles were also used to compare
candidates, so the CV results are development estimates, not independent
holdout scores; the private scores came out 0.6 to 1.7 points lower.
Errors concentrate in close calls: for `tactical_team_spikes`, the 25.5% of
battles predicted within 10 points of 50/50 are right 53.0% of the time,
against 89.4% for the 28.0% predicted more than 30 points away
([details](docs/development_history.md#22-close-calls)).

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
| Corrected features | Four feature-code fixes | 0.7100 |
| Similar-team stage | Logistic regression on the corrected mixture | **0.7118** |

The first seven rows use the original seed-42 CV. The next three are means of
three grouped partitions. The last three score averaged OOF probabilities; their
separate-partition means are 0.7060, 0.7076 and 0.7080. These are historical milestones with different
evaluation procedures, not one directly comparable score series. The team-record
row is after its tie-breaking fix; its earlier submitted file used the previous version.

Trajectory and basic type summaries, hierarchy/CatBoost mixtures, uncertainty
features, ordered observations, state-signal trials and learned mixture weights
are also documented, including cases that did not justify a replacement.

### Experiment branches

Each branch keeps the code and log as they stood at one step, so an
experiment can be read or rerun in its own context. `main` holds the full
history and the code that reproduces the three selected submissions.

| Branch | What it holds |
|---|---|
| `e06-model-comparison` | Starter template, E00 to E06 experiments and the experiment log |
| `model-checks` | Wide LightGBM/XGBoost blend, untouched-seed checks, tuning and CatBoost checks |
| `data-limits` | Limits analysis, the corrected reading of the faint counts, Gen 2 mechanics tables |
| `compact-model` | Gen 2 role counts and the 176-column permutation-importance selection |
| `team-records` | Recurring teams and their cross-fitted records |
| `win-conditions-v2` | Unanswered Pokémon from a learned matchup chart |
| `battle-state-reconstruction` | Reconstructed battle state and the audit fixes |
| `corrected-features` | The four feature corrections as one change |
| `decision-quality-features` | Move-quality features from a turn-by-turn value model |
| `final-reproduction` | Code that rebuilds the three selected submissions |

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

**Feature corrections.** `--feature-fixes` corrects the move accuracy scale,
calibrates the transferred P2 knockout scores, computes knockout risk for every
P2 species and gives validation rows inner estimators of the same size as
training rows. Without it the code reproduces the original submission exactly.

**Second stage.** A logistic regression takes the logit of the tree probability
and the win rates of similar P1 and P2 teams, optionally with whether Spikes is
still down on each side at turn 30. It is fitted on the out-of-fold rows of the
three partitions. Applied to the tactical view alone, it scores about as well as the
corrected five-view mixture (mean partition accuracy 0.7077 against 0.7076).

[Methodology](docs/methodology.md) explains the features, approximations, inner
splits and results. [Reference results](results/reference.json) record the
expected metrics and prediction hashes.

## Reproduce

Use Python 3.12 (tested with 3.12.14) and the versions in `requirements.txt`. Place the four competition
files in `data/`: `train.jsonl`, `test.jsonl`, `sample_submission.csv` and
`type_chart_gen2.csv`.

```bash
python -m pip install -r requirements.txt

# 1. Original five views
python run.py predict --output outputs/original

# 2. Corrected five views, then the similar-team stage
python run.py predict --feature-fixes --output outputs/fixed
python run.py stage --output outputs/fixed --view blend --signals team --name team_similarity

# 3. Corrected tactical view with similar teams and Spikes (reuses run 2)
python run.py stage --output outputs/fixed --view tactical --signals team,spikes --name tactical_team_spikes
```

Each `predict` builds every feature from the competition files, trains all
fifteen fold ensembles, evaluates OOF predictions and writes a submission plus
per-fold probabilities, logs and `metrics.json`. It takes about two and a half
hours on two cores. `stage` takes a few minutes. The expected SHA-256 of each file is in
[selected_submissions.json](results/selected_submissions.json), and the files
themselves are in [results/submitted](results/submitted). Two workers are the
default; use `--jobs 1` if memory is limited.

On our machine the rebuilt files match the submitted ones byte for byte. Tree
models are sensitive to floating-point differences between processors, so on
other hardware a few borderline predictions can flip. On Kaggle the notebook's
rebuilt files agree with the submitted ones on 4,968, 4,964 and 4,922 of the
5,000 test battles (99.4%, 99.3% and 98.4%), and pooled CV stays within 0.12
points (0.7095, 0.7106 and 0.7085 against 0.7103, 0.7118 and 0.7085).

To run CV without loading the test set:

```bash
python run.py cv --jobs 2
python -m unittest discover -s tests -v
```

Interrupted runs reuse completed folds only when the input files, source and
package versions match. Use a new `--output` directory after changing any of
them. To rebuild the CSV and metrics from completed folds:

```bash
python run.py collect --output outputs/original
```

## Code layout

- `run.py` is the only training entry point.
- `src/pipeline.py` prepares inputs, runs folds and writes predictions.
- `src/validation.py` holds splits, routing, weights and the threshold.
- `src/features.py` assembles the four core views and their inner estimates.
- `src/options.py` holds the `--feature-fixes` switch; `src/second_stage.py` fits the similar-team and Spikes stage.
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
