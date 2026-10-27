# State reconstruction and candidate integration

The frozen Kaggle candidate is the equal LightGBM/XGBoost state-reconstruction blend: **70.1300% mean accuracy** over three repeated five-fold grouped CV partitions. The three earlier candidates' distinct components were integrated and tested; none passed the predefined guard for replacing the state model. The submitted file scored **0.69533** on the public leaderboard on 27 October 2026, placing the team fourth.

This experiment builds on `battle-state-fixes` (`0bffd558fbe82525636eccf7bd30ec791267783e`) and its win-conditions-v2 ancestry. It includes the three original fixes: no artificial zero-damage hit for empty attacks, inclusion of the known P2 lead in roster matching, and no resurrection of an observed-fainted P2 species. The evaluated source modules are pinned by Git blob hash. The earlier training entry point remains available; the Kaggle notebook now runs this model.

## Components integrated and tested

| Earlier candidate | Work already retained in state reconstruction | Distinct part tested here |
|---|---|---|
| Submission 01: baseline/combined representation blend | Corrected baseline feature families, observation features, move-conditioned opponent roster posterior, relative threats, LightGBM and XGBoost | Adding its older model probabilities at 25% or 50% |
| Submission 02: combined-feature blend | All its feature groups and the same model parameters | Adding its simpler representation at 25% or 50% |
| Submission 03: baseline/CatBoost blend | Its shared observation, roster and threat features | Eight hierarchical item/lead/team-strength columns, ten categorical team/active-species/move fields, and CatBoost retrained with reconstructed-state features |

Shared groups appear once. Replacing the 15-column posterior block with its 23-column hierarchy superset adds the eight numeric columns without duplicates. CatBoost retains the earlier parameters: 700 iterations, depth 5, learning rate 0.04 and regularization 10. No new hyperparameter or threshold search was used.

The selected state model adds possible unrecorded P2 KO evidence, remaining HP distribution and recovery resources, departure/switch evidence, and cross-fitted species/move/team records. The auxiliary event classifier transfers evidence from observed P1 events; its P2 scores are uncertain features, not measured or calibrated death probabilities. [The historical feature study](deep_features/README.md) explains their derivation; use the results below for the corrected implementation.

## Reproducibility fixes

Two problems appeared when reproducing the earlier scratchboard submission:

1. Unordered move sets changed floating-point posterior accumulation. Near-tied roster candidates could cross the top-three cutoff. Canonical move/species iteration and a deterministic tie-break now stabilize selection. Only the ranking key is rounded, not the posterior weights.
2. Reading unknown moves from `defaultdict(set)` inserted empty observations. Learned prior populations could then depend on feature construction order and cache reuse. Observed-move maps are now ordinary dictionaries, reads are non-mutating, and empty evidence is excluded from learned move-prior populations.

Every affected model family was recomputed on the same three grouped CV partitions. Unchanged baseline matrices and predictions were verified and reused. The older **70.2267%** score and `submission_state_reconstruction.csv` precede these fixes and are historical; the corrected branch has a newly generated submission.

## Matched CV results

All scores use the same 10,000 training battles. The three split seeds are repeated partitions of the same outcomes, not independent untouched validation sets. Classification uses threshold 0.5 throughout. No public leaderboard feedback or test probabilities selected the models.

| Representation / mixture | 20261026 | 20261105 | 193 | Mean accuracy |
|---|---:|---:|---:|---:|
| Previous submission 01 | 69.12% | 69.37% | 69.48% | 69.3233% |
| Previous submission 02 | 69.30% | 69.28% | 69.08% | 69.2200% |
| Previous submission 03 | 69.08% | 69.43% | 69.07% | 69.1933% |
| **Selected state model** | 70.20% | 69.95% | 70.24% | 70.1300% |
| 75% state + 25% previous 03 | 70.45% | 70.27% | 69.98% | 70.2333% |
| 75% state + 25% previous 01 | 70.32% | 70.30% | 70.03% | 70.2167% |
| State features + hierarchy, LGB/XGB | 70.13% | 70.07% | 70.30% | 70.1667% |
| Half state / half hierarchy blend | 70.18% | 69.97% | 70.33% | 70.1600% |
| State + 25% new full-feature CatBoost | 70.21% | 69.96% | 70.19% | 70.1200% |
| State + one-third new full-feature CatBoost | 70.19% | 69.93% | 70.18% | 70.1000% |
| Hierarchy LGB / XGB / CatBoost, equal thirds | 70.07% | 70.07% | 70.33% | 70.1567% |

The numerically highest integration is 75% state plus 25% previous 03: **70.2333%**, a gain of **0.1033 percentage points**. Its per-seed changes are +0.25, +0.32 and **-0.26** percentage points. It fails the stability guard. The hierarchy blend gains only **0.0367 points**; its exploratory paired 95% interval is **-0.1933 to +0.2666 points**. Neither provides a reliable reason to add components to the submission.

The selection guard requires at least +0.10 mean percentage points, gains in at least two seeds, no loss exceeding 0.10 points in any seed, and lower mean log loss. None of the fixed integrations passes all conditions. CatBoost improves log loss in some mixtures without improving accuracy, the competition metric. All tested implementations and metrics remain in this branch, while `merge_check/selection.json` enables only the selected state blend.

The selected model improves over the corrected previous 01 mean of 69.3233% by **0.8067 percentage points**, with +108, +58 and +76 net correct predictions per 10,000 across the three seeds. The exploratory paired battle-cluster bootstrap interval is **+0.2567 to +1.3341 points**. Repeated seeds remain together within each sampled battle cluster; these are not 30,000 independent battles. The interval is not adjusted for repeated model/feature selection and does not guarantee a Kaggle gain.

## Reproduce the submission

Use Python 3.12 and place the supplied challenge files in `data/` at the repository root:

```bash
python -m pip install -r experiments/state_reconstruction/requirements.txt
python experiments/state_reconstruction/run.py predict --data data
```

This writes `submissions/submission_state_reconstruction_deterministic.csv` and does not submit to Kaggle. `--output /path/to/file.csv` selects another destination. Final prediction verifies the supplied data hashes. A source ZIP or shallow checkout also works if its reference source files match the pin exactly. Runtime package versions are pinned in `requirements.txt`.

`deep_features/results/submission_manifest.json` records the data and source hashes, feature count, class counts and submission hash. `reproduction_check.json` compares a fresh full fit with a cached fit under different Python hash seeds, including complete feature matrices, probabilities and submission bytes.

## Reproduce the comparison

```bash
python experiments/state_reconstruction/run.py cv --data data
```

Use `--seeds 20261026` for a single partition. The complete three-partition run performs grouped-fold checks, paired comparisons and a duplicate-prefix sensitivity check. Generated caches and OOF/checkpoint files are reused. If changing feature/model code, remove the corresponding derived caches, `.npz` and `.npy` checkpoints before rerunning under the same names; tracked JSON/CSV metric records alone do not skip training.

After the full CV run, recompute the comparison without retraining:

```bash
python experiments/state_reconstruction/run.py review --data data
```

The review writes a recommendation; the frozen prediction weights remain explicit in `merge_check/selection.json`. `merge_check/results/summary.csv` contains every fixed mixture, `review.json` explains the guard, and `selected_validation.json` contains the corrected selected model's paired comparisons. `deep_features/results/validation.json` records the grouped-fold and duplicate-prefix checks. Raw data, caches and generated submissions are excluded from version control.

Additional checks:

```bash
python -m unittest discover -s tests -v
python experiments/state_reconstruction/check_determinism.py --data data
```

The six regression tests cover the original fixes, roster tie-breaking, input-order invariance and non-mutating feature reads. The hash-seed check reproduces 90 context columns across three processes and checks that feature extraction does not alter observed move maps. `cache_equivalence.json` records equality of freshly computed versus refreshed train/validation blocks.

## Validation boundaries

- Outer and inner fitting separates exact-timeline groups. Winner-derived features are cross-fitted within outer-training data. Split seeds and model parameters are fixed.
- The auxiliary model uses only within-window P1 event labels. Training-row event scores exclude their battle and duplicate group via another inner split.
- Removing 56 duplicate-prefix rows from scoring preserves the state model's improvement. These sensitivity checks do not make model selection independent of the training data.
- Only the supplied battle data and type chart are used. No external replay data, usage tables, move databases, pretrained models or neural networks are added.
- The historical feature-study README preserves earlier exploratory findings. Its old metrics and submission hash are not the current branch result.
- This model is now included in main. The public score is one check on the frozen candidate; it was not used to select its features or weights.
