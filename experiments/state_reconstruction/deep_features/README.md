> Historical feature study before the deterministic roster-selection and non-mutating feature-read fixes. The scores and CSV hash below describe that earlier run. Use [the current study README](../README.md) and its frozen selection for the branch candidate.

# Battle-state reconstruction experiments

This scratchboard extends the earlier experiments. It recommends **one new submission**, `submission_state_reconstruction.csv`. It does not merge changes into main, edit the course report, or submit to Kaggle.

The selected model averages LightGBM and XGBoost probabilities with fixed 0.5/0.5 weights and uses a 0.5 classification threshold. Model parameters are unchanged from the earlier scratchboard. The new work is feature engineering, including an auxiliary classical classifier that turns observed within-window battle events into additional features.

## Results

Accuracy below is out-of-fold accuracy on the supplied 10,000 training battles, not a Kaggle score. All rows use the same three sets of five outer folds. Seed 20261026 was the feature-screening split; the two other seeds checked sensitivity to the partition. These are repeated partitions of the same battles, not three independent datasets.

| Model / representation | 20261026 | 20261105 | 193 | Mean |
|---|---:|---:|---:|---:|
| Corrected existing team model | 69.13% | 69.12% | 68.80% | 69.02% |
| Previous submission 01, its fixed final weights | 69.21% | 69.44% | 69.24% | 69.30% |
| Previous combined-feature model | 69.25% | 69.51% | 69.22% | 69.33% |
| Previous combined + resource/departure/switch features | 69.51% | 69.46% | 69.51% | 69.49% |
| Previous combined + auxiliary KO evidence | 69.79% | 69.74% | 69.63% | 69.72% |
| Previous combined + both new groups | 69.94% | 69.67% | 69.86% | 69.82% |
| Above + cross-fitted species/move/team records: selected | **70.35%** | **69.91%** | **70.42%** | **70.23%** |

The selected candidate improves on previous submission 01 by 114, 47, and 118 correct predictions per 10,000, respectively. Mean gain is **0.93 percentage points**. Mean log loss falls from 0.57012 to 0.55703. It wins in 13 of 15 matched folds, ties in one, and loses in one; these folds are correlated and are not independent statistical replications.

A paired bootstrap resamples exact-timeline battle clusters, keeping all three seed results together within each battle. Its exploratory 95% interval for the accuracy gain is **+0.386 to +1.466 percentage points** against previous submission 01, and **+0.607 to +1.790** against the corrected existing model. These intervals do not account for choosing features/models after inspecting CV. They are evidence of a stronger local signal than the earlier candidates, not a guarantee of public/private leaderboard improvement.

Adding the species/move/team record block to the reconstructed-state model improves all three seeds, by 41, 24, and 56 correct predictions. Its exploratory paired interval is +0.067 to +0.744 percentage points. This is a conditional comparison within this representation, not a causal attribution or proof that these features help every model.

The original seed-42 score of 0.6947 is not used as a directly comparable baseline here. The corrected existing model retains the original global label-free team catalog and inner StratifiedKFold encoding; its outer folds match the new models. The new models rebuild learned resources on outer-training data and group their inner encodings. The stricter baseline is also retained in the results (`strict_base`: mean 68.993%).

## The missing signals

### 1. A disappearing opponent is not necessarily a living reserve

Using the last recorded HP/status literally can preserve a Pokemon that actually fainted between the snapshots. The data contain 16,726 transitions into recorded P1 fainted status versus 4,976 for P2. This is evidence of asymmetric observation; it does not prove that every unmarked P2 departure was a KO.

The new auxiliary classifier learns from P1 events inside the visible 30-turn window. Its positive examples are observed same-species alive-to-fainted transitions, excluding recorded self-explosion and examples without a later incoming Pokemon inside the window. Its negative proxy examples are unmarked departures that do not return within the window. These negatives can contain label noise.

Inputs include the departing Pokemon's previous HP/status, observed attack and type matchup, boosts, opponent HP changes, and the incoming Pokemon's HP relative to its previous observation. The incoming HP matters: a regular switch exposed to an attack and a replacement after a knockout can leave different traces. Type immunity, entry damage, recovery and snapshot timing prevent treating this as a hard rule.

Apply the model to unmarked P2 departures and aggregate soft evidence: expected missing count, HP discounted by risk, high-risk count, maximum risk, likely missing recovery users, and species-specific adjusted HP. Original observed features remain available. No original Pokemon is declared dead or removed as a fact.

The auxiliary classifier gets roughly 97% accuracy on the held-out **P1 proxy event task** after adding incoming-HP evidence. This is **not measured P2 hidden-KO accuracy**, and its scores are not calibrated P2 death probabilities. Transfer to P2 is supported only by downstream winner-prediction CV. Own-battle and duplicate-timeline events are excluded from training features through inner cross-fitting.

### 2. Equal total HP can represent very different remaining resources

Sorted individual HP, HP concentration/entropy, healthy and near-fainted counts distinguish a few useful survivors from many nearly depleted Pokemon. Observed recovery access and Rest/Sleep Talk combinations distinguish recoverable damage from more persistent loss. The representation also tracks low-HP hazard/removal/recovery users, status freshness, active sleep/effect streaks, and HP-weighted boosts.

Switch features add relative matchup improvement, immunity/resistance entries, entry HP loss, double switches, phazing observations and approximate hazard evidence. Damage pressure is a heuristic based on supplied base stats, observed move metadata and the provided type chart; it is not an exact Gen 2 damage simulator. Hazard variables are soft observations, not a reconstructed authoritative field state.

These hand-crafted blocks alone produce a small gain. Their combination with learned missing-KO evidence is stronger. Training-set feature importances also rank HP concentration, healthy count and discounted opponent HP highly; those importances are descriptive, not an independent validation test.

### 3. Six species do not uniquely describe a recurring team

The same species roster can expose different move combinations. `MoveRecords` learns smoothed training-only records for species/move tokens, pairs of observed moves, and roster/species/move combinations. It aggregates support, spread, extreme records and HP-weighted evidence. Ambiguous P2 rosters are averaged over the top three compatible hypotheses from the prior roster model.

Every training row's target-derived records exclude its own inner fold and exact-timeline duplicates. Outer-validation records use only outer-training outcomes. This block may encode recurring sets, player population differences, or other correlations; without player identities it cannot be interpreted purely as strategic set strength.

## Rejected or limited avenues

- Explicit roster-order features did not improve the screening split. Most supplied order follows observed reveal order, so treating it as a planned initial lineup would also be questionable.
- Items, levels and HP precision have unusual values but their additional feature block did not establish an improvement. Rare-category marginal win rates have very small support.
- Basic departure/low-HP flags and isolated resource features were weak. Incoming-HP evidence and the auxiliary classifier made the hidden-state hypothesis more useful.
- Hidden Power's supplied type/category/power fields remain unresolved. No external move table or invented replacement power is used.
- Repeated move observations are preserved, with additional observation flags. Their pattern is compatible with persistent last-move snapshots, but the actual extraction code was not available; this is not asserted as a proven schema bug.
- PP counting, exact sleep counters, complete field reconstruction, and bench-status correction cannot be recovered reliably from these snapshots alone. Current clocks are observed streaks, not exact engine counters.

## Validation and boundaries

All 15 outer folds separate identical timeline groups. Auxiliary event training uses two inner grouped folds (seed 347), and target-derived team/move records use five inner grouped folds (seed 99). Vocabularies, roster catalogs, species metadata and learned records are based on outer-training battles. Supplied move/type/stat information is used; no external dataset, replay, usage table, pretrained model or neural network is used.

An audit of side-swap-invariant first-10-turn duplicates finds 56 affected rows. Excluding all 56 from scoring leaves selected accuracies of 70.394%, 69.942%, and 70.444%, versus 69.228%, 69.479%, and 69.248% for previous submission 01. This is a scoring sensitivity check, not a retraining with every prefix grouped.

On 80 sampled battles, changing the winner label and battle ID leaves all new label-free features and auxiliary event labels unchanged. Feature matrices omit both fields, have unique columns and finite values. This complements the fitting-boundary checks; it is not a proof against every possible source of leakage.

The final feature family, model weights and threshold were frozen before constructing the new test features/predictions. The test data had been inspected in earlier project auditing but were not used to select this candidate. The final CSV has exactly 5,000 unique sample-matching IDs, two required columns and binary labels. There are 2,576 predicted P1 wins. SHA-256: `0f527b486abe92ed92887422e167c01b5d04dd1bd8849686011f6216ac3c77a8`.

## Reproduce

Extract the bundle with sibling directories `scratchboard/`, `deep_features/` and `fix_impact_check/`. Put the competition files in sibling `upload/`: train.jsonl, test.jsonl, type_chart_gen2.csv and sample_submission.csv (the supplied filename containing `(1)` also works). Raw data, feature caches and copies of the GitHub repository are intentionally omitted.

Python 3.12 and the pinned CPU packages in `scratchboard/requirements.txt` were used. Read the pinned source from an existing repository checkout; this does not change that checkout:

```bash
python -m pip install -r scratchboard/requirements.txt
python scratchboard/bootstrap_reference.py --repo /path/to/FDS-Pokemon-Challenge-2026-2027
```

The reference pin is fix commit `0bffd558fbe82525636eccf7bd30ec791267783e`; original main is `dc81b84d138c0f374333c78fb0e88134daa28972`. The fixes branch has a win-conditions-v2 parent. Integration should review/cherry-pick the intended changes rather than assuming the whole branch contains only fixes.

To reproduce the selected candidate on all three split seeds, run the following for each value of `S` in 20261026, 20261105 and 193:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scratchboard/prepare.py --data upload --seed S
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scratchboard/experiment.py --seed S --models base_lgb base_xgb combined_lgb combined_xgb
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/prepare.py --seed S
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/auxiliary.py --seed S --entry
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/move_records.py --seed S
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/experiment.py --seed S --models prior_state_lgb prior_state_xgb prior_entryko_lgb prior_entryko_xgb deep_entry_lgb deep_entry_xgb deep_entry_sets_lgb deep_entry_sets_xgb
```

Then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/corrected_baseline.py
python deep_features/summarize.py 20261026 20261105 193
OPENBLAS_NUM_THREADS=1 python deep_features/validate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python deep_features/predict.py
```

Scripts reuse existing caches and result files. To actually retrain bundled results, move the relevant `.npz`/`.json` result files aside first; do not change feature code and silently reuse an old cache under the same name. `predict.py --rebuild-base` explicitly rebuilds the older full-training feature cache. Its new feature groups are always rebuilt.

The broad first-split screening results are included in `deep_features/results/`. Reproducing the non-entry auxiliary variant additionally needs `auxiliary.py --seed 20261026` without `--entry`. Avoid the default all-model experiment list unless all required caches have been prepared.

`selection.json` freezes the one recommended candidate. `results/validation.json` and `paired_folds.csv` contain paired comparisons and checks; `summary.csv` contains the full screen. The `.npz` files retain OOF probabilities. `results/submission_manifest.json` records data, source and submission hashes. The probability and feature-importance CSVs are diagnostics, not Kaggle submissions.

## Next decision

Use one Kaggle attempt for `submission_state_reconstruction.csv` before spending attempts on nearby mixtures. Keep the current leaderboard submission available for comparison. Judge the fixed candidate after observing its score; do not tune thresholds from public-score feedback. The local gain is promising, but a public score is noisy and the hidden partition may differ. Main and the course report remain unchanged until the chosen work is integrated.
