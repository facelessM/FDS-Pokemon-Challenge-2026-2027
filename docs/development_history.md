# Development history

We started with team averages, moved to the observed battle state, and then tested which information that representation was losing. Some additions became part of the selected model. Others were useful checks but did not justify a replacement. The final improvement came from combining different representations and matching the submission procedure to the evaluated predictor.

The numerical record is in [history.json](../results/history.json). The current method is described in [methodology.md](methodology.md). Earlier executable code remains in the [checkpoint before the main cleanup](https://github.com/facelessM/FDS-Pokemon-Challenge-2026-2027/tree/f4c233b8e9523779b3a29c8eec1e3d229323f158).

## How to read the scores

The first experiments used stratified five-fold CV with seed 42. Later checks repeated that procedure with seeds 6 to 10. State reconstruction introduced three grouped five-fold partitions, with seeds 20261026, 20261105 and 193, so identical timelines could not appear on both sides of a split.

The current result also averages each battle's three held-out probabilities before applying 0.5. Its 0.7103 accuracy is different from the 0.7060 mean of the three separate partition accuracies. Scores from different rows below are not automatically paired comparisons. Extra seeds still reuse the same battles and do not create an independent test set. Historical numbers are retained as recorded; they have not all been rerun through the current code.

## 1 Starter baseline

E00 used six team/lead averages and logistic regression. Accuracy was 0.5178. Scaling those inputs gave 0.5164. This established that team averages alone were weak predictors and that scaling did not explain the next improvement.

## 2 Last observed battle state

E01 added last-observed HP, visible faints, status, active boosts and side differences. The 39-column model reached 0.6671 and became the first submitted approach. Team HP difference alone reached 0.6519, showing where much of the early signal came from.

We initially treated the last observation as an accurate turn-30 state and read P1's excess recorded faints as evidence that P1 was usually behind. Later checks changed both interpretations. Bench observations can be stale, and ordinary P2 attack knockouts are often unmarked. The selected method keeps observation age and uncertain departure evidence instead of treating those assumptions as recovered facts.

## 3 Trajectory and simple type matchups

E02 added HP snapshots, slopes, lead changes and late damage. Accuracy fell to 0.6657. Much of that representation correlated with the final HP difference. Default gradient boosting also showed no gain from that particular addition. We did not retain it as a separate feature group; this did not establish that all temporal information was useless.

E03 added base-stat summaries and type advantages using the supplied chart and species metadata recovered from training battles. Accuracy was 0.6665. The type calculation used species types and could not represent unrevealed coverage moves. We left the group out of the main feature set and later returned to matchups with move-set hypotheses and remaining resources.

## 4 Error analysis and sacrifice moves

E04 examined out-of-fold errors, close predictions, status, remaining species and behaviour counts. It also exposed the need to distinguish Explosion and Self-Destruct from other faint evidence.

E05 made that distinction with six sacrifice features. Logistic regression reached 0.6712, against 0.6671 for E01. Default gradient boosting did not improve: it scored 0.6695 with the added features versus 0.6707 on E01. The benefit depended on the model, so we kept the interpretation without assuming every classifier would gain equally.

## 5 Model comparison and the first wide feature set

With E05 fixed, logistic regression scored 0.6712, random forest and default gradient boosting both 0.6695, and scaled k-nearest neighbours 0.6426. We then added move counts, observed move effects and species still standing. These sparse features worked better with boosted trees than with the linear model.

The 404-column LightGBM/XGBoost blend reached 0.6875 on seed 42 and 0.6841 averaged over seeds 6 to 10. The corresponding E05 comparison on those repeated seeds was 0.6690. LightGBM alone scored 0.6878 on seed 42 but 0.6820 on seeds 6 to 10. We retained the blend for its repeated-partition results. A 20-setting LightGBM search did not improve the selected configuration.

This was the next submitted approach. Explicit mechanics summaries and a simple hidden-KO correction were also checked, but neither improved the existing model sufficiently. Those results did not prove that the mechanics were already represented completely; later work used more specific state and move evidence.

## 6 Compact model

We added Spikes, Rapid Spin and Explosion role counts, then used permutation importance to retain 176 of 413 columns. The compact blend scored 0.6885 on seed 42 and 0.6842 on seeds 6 to 10, nearly the same as the wider blend's 0.6841. It was submitted as a smaller alternative. Feature selection had used these battles, so its seed-42 score was not an independent check.

## 7 Team records and deterministic matching

Many team compositions recur. We added smoothed P1 team records and records for compatible P2 teams, calculated with inner folds so a training row did not receive its own outcome. The first logged version scored 0.6933 on seed 42. A tie-breaking fix changed the result to 0.6947; the corrected model averaged 0.6901 over seeds 6 to 10.

The original matching code could choose whichever tied roster appeared first in a Python set. Canonical ordering made that choice repeatable. The submitted `team_blend.csv` preceded this fix, so its predictions must not be described as the corrected CV run. Team records can also carry recurring-player effects, but player identity is not available to separate that from composition.

## 8 First win-condition model

We learned a species-pair chart from observed net HP exchanges and asked whether a remaining Pokémon had a favourable worst matchup against the opposing survivors. `wincon_blend.csv` scored 0.6939 on seed 42 and 0.6929 averaged over seeds 6 to 10, compared with 0.6901 for team records on those repeated seeds.

The idea remained useful, but the chart pooled different move sets, statuses and switching situations. It also depended on inferred survivors. The later tactical model revisited the question using consistent move hypotheses, defensive coverage, recovery and trades. We retained the original experiment as an earlier approximation, rather than treating its chart as a solved matchup model.

## 9 State reconstruction and recording fixes

We separated HP by recovery access, added switch and departure evidence, learned uncertain P2 knockout scores from recorded P1 events, and added species-move records. Identical timeline groups stayed together in validation, and outcome-derived records used inner grouped splits.

The audit fixed empty damage counts, opponent-lead matching, fainted members being reintroduced by roster hypotheses, unstable roster ties and move lookups that changed dictionaries while reading them. Affected representations were recomputed. Unchanged reference predictions were reused only when their source, features and folds were unchanged. Later cache signatures made those dependencies explicit.

The selected state model scored 0.7020, 0.6995 and 0.7024 on the three grouped partitions, for a mean of 0.7013. The earlier candidate averaged 0.6932 on those same splits. Hierarchy features and CatBoost mixtures did not produce a consistent further gain. This stage produced `submission_state_reconstruction_deterministic.csv`.

## 10 Additional uncertainty features

We tested KO-risk weighting in matchups, using all compatible P2 rosters, and additional evidence about repeated moves, entries, Heal Bell and hazard timing. Their mean grouped accuracies were 0.70090, 0.70070 and 0.70117; combining all three gave 0.70000, against 0.70130 for the state reference. None passed the recorded screen, so no replacement submission came from this round.

## 11 Species state and forecasts

Team totals did not fully describe which species held the remaining HP. We tested 320 species-state columns, 288 species-move resource columns and nine five-turn forecast features. Their separate means were 0.70303, 0.70137 and 0.70207. Combining all three gave 0.70213.

The follow-up kept species state and forecasts but dropped the new species-move resource block. It reached 0.70360 across the three grouped partitions. On the additional partition 20261029, it scored 0.7071 against 0.7024 for the reference. This produced `submission_species_forecast.csv`. The forecast learned changes within the visible timeline; it did not use turns after 30.

## 12 Tactical coverage

We returned to win conditions with consistent partial move sets, remaining defenders, backup answers, recovery, hazards and Explosion trades. The chosen joint view scored 0.70513 across the three grouped partitions with model seed 0.

Across nine partition/model-seed comparisons, its mean was 0.70391, a gain of 0.00310 over matched references, with eight positive comparisons. The extra partition scored 0.7048 against 0.7024. These checks supported `submission_tactical_coverage.csv`; they were repeated evaluations of the same battles, not independent evidence that every gain would transfer.

## 13 Fundamental checks and alternative representations

We checked whether increasingly elaborate features were masking a simpler problem. On one existing partition, removing outcome records lowered accuracy from 0.7020 to 0.6948. An observed-block model scored 0.6901 and a compact resource summary 0.6718. This supported retaining the reference and looking for complementary information, without establishing a confirmed source of unseen-data differences.

Ordered timelines, active stints and raw summaries supplied different representations. An inner-selected complementary model scored 0.7029 against 0.7024 on partition 193, but worsened log loss and failed its primary screen. Calibration alone scored 0.7001. We did not promote that standalone procedure. Raw summaries and stints were still available for a later, separately recorded mixture test.

## 14 Isolated state-signal trials

We tested softer KO treatment, possible Heal Bell updates and learned pressure for damaging moves with zero recorded base power. Mean grouped accuracies were 0.70183, 0.70047 and 0.70223. Their joint version scored 0.70140 against the 0.70130 reference. None passed the prespecified replacement guard.

Zero-power pressure nevertheless improved accuracy on all three partitions and made some different errors. We kept it as a candidate component for the next combination study, without relabelling the isolated test as a clear standalone improvement.

## 15 Combining the views

Appending the new feature groups into one model gave 0.70170 mean accuracy. Averaging the four core views gave 0.70400. Adding raw summaries gave 0.70600; adding stint summaries instead gave 0.70640. Separately fitted views therefore worked better here than the direct feature union.

We also fitted mixture weights inside the outer training folds. On partition 193, that candidate scored 0.7066 versus 0.7062 for the fixed summary mixture. Its paired interval included zero, and the primary gate failed. We retained fixed weights and did not present the learned-weight candidate as validated.

## 16 Matching the selected CV and submission

A secondary check averaged each battle's three held-out probabilities. The state reference reached 0.7026; the four-core mixture 0.7043; the core-plus-stints mixture 0.7092; and the core-plus-raw-summary mixture 0.7103. The last became the selected predictor. Its mean separate-partition accuracy remained 0.7060.

To submit that predictor, we kept the 8,000-row training folds and gave each test battle one fold ensemble per partition. Known timeline groups used their held-out folds; other groups used fixed label-free hashing. The three probabilities were averaged at the same 0.5 threshold. This produced `submission_matched_core_summary.csv`.

The main cleanup changed organization, not the selected prediction rule. All fifteen fold ensembles were trained again. Every component probability, the combined OOF and test probabilities, and the final 5,000-row CSV matched exactly. Nine regression tests passed. This verifies reproduction; the 0.7103 result remains a development estimate selected after many experiments on the same battles.

## 17 Late momentum, boost scaling, soft roster completion and full bagging trials

We subsequently evaluated three candidate directions aimed at addressing tactical momentum and Player 2 unrevealed roster uncertainty:

1. **Effective HP and Late-Prefix Momentum:** We evaluated exact Gen 2 stat multipliers stages(s) on Defense and Special Defense, complete setup move tracking (Curse, Swords Dance, Growth, Belly Drum, Agility, Amnesia), and late-prefix switch counts and damage taken (turns 20–30) in resources() and switches(). On Fold 0, accuracy rose from 0.7155 to 0.7160. Across all 10,000 battles across the 15 folds, averaged OOF accuracy was 0.7108 (+0.0005 over the 0.7103 reference, +5 net correct battles) with log loss improving from 0.55301 to 0.55284. Because the gain (+0.0005) fell within random sampling variance and failed the replacement guard, the production entry point was kept in its clean, frozen state.
2. **Soft Roster Completion:** We tested continuous imputation of Player 2 unrevealed rosters using an empirical co-occurrence matrix and Gram partner substitutability (S = C_norm @ C_norm.T). On Fold 0, accuracy dropped from 0.7155 to 0.7130 (-0.0025). The continuous imputation removes the clean categorical threshold boundary that tree-based models leverage to isolate battles with unobserved opponents. The mathematical formulation was retained offline in analysis/cooccurrence_analysis.py.
3. **Variance Weighting and Non-Linear Aggregation:** Using out-of-fold probability distributions, we evaluated Gauss-Markov inverse-Brier weighting, Bernoulli uncertainty weighting, trimmed mean, and median voting in analysis/confidence_variance_study.py. Robust non-linear aggregations reduced accuracy (median voting scored 0.7040 vs 0.7072 equal weighting) by suppressing specialist model signals, while inverse-Brier weights yielded negligible difference (+0.0004). Uniform equal weights were therefore reaffirmed.

Following the frozen reference principle, the production training code was left unchanged and the analytical scripts were preserved in analysis/.

## Submitted approaches in order

| Order | Approach or file | Development stage |
|---|---|---|
| 1 | Turn-30 state model | E01 |
| 2 | First wide LightGBM/XGBoost blend | Move usage, effects and surviving species |
| 3 | `compact_blend.csv` | Role counts and feature selection |
| 4 | `team_blend.csv` | Team records, before the tie-breaking correction |
| 5 | `wincon_blend.csv` | Species-pair exchange chart |
| 6 | `submission_state_reconstruction_deterministic.csv` | Reconstructed resources and recording fixes |
| 7 | `submission_species_forecast.csv` | Species state and five-turn forecasts |
| 8 | `submission_tactical_coverage.csv` | Move hypotheses and defensive coverage |
| 9 | `submission_matched_core_summary.csv` | Five views with matching three-partition prediction |

Further experiments should extend this record with their hypothesis, evaluation procedure, result and decision. The selected training entry point remains separate from the experiment history.
