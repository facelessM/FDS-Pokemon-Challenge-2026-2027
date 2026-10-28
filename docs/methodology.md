# FDS Pokémon Battle Prediction Methodology Report

Oct 28, 2026 · Özgür Cihan Karahan

## 1 Team and task

Group: A wild Gengar Appears!

| Name | Student ID |
|---|---|
| Özgür Cihan Karahan | 2233702 |
| Gabriele Fiacconi | 2060444 |
| Musa Alasgarov | 2242081 |
| Baran Berkay Hökelek | 2243542 |

We predict whether Player 1 wins a Gen 2 OU battle from its full team, Player 2's lead and the first 30 active turns. Battles last between 30 and 80 turns, and their total length is hidden. The target is the final outcome. Accuracy is the competition metric, and only classical machine learning is allowed.

Our selected model combines five views of the same battle. We average their probabilities within each fold, then average three out-of-fold predictions per battle. This gives 0.7103 accuracy, compared with 0.7026 for the earlier state model evaluated in the same way.

Code: github.com/facelessM/FDS-Pokemon-Challenge-2026-2027

## 2 Data and recording limits

There are 10,000 labelled training battles and 5,000 test battles. Each timeline has 30 entries containing the active species, HP fraction, status, effects, boosts and recorded move details for both sides. Player 1's six species are known from the start. Player 2's other members and moves become visible only when recorded in the timeline.

We retain the last observed state of each Pokémon. Bench HP and status can become stale, so the features also record how long ago a Pokémon was seen. Unrevealed members are assigned full HP in team-total proxies; this is an assumption used by the representation, not a recovered observation. Explicit faints remain fainted when a possible opponent roster is added.

The recording is asymmetric. P1 attack knockouts can appear with status `fnt`; many P2 departures have no such marker. A Pokémon that leaves and never returns may have switched or fainted. We therefore keep departure evidence separate from confirmed faints. Likewise, a non-null move field is a recorded move, not sufficient proof of a fresh action at that exact transition. Raw repetition counts remain available, while learned HP-response features use more restrictive intervals.

The training data contains 9,983 distinct timeline hashes. We group identical timelines together even if their battle IDs differ. `battle_id` is used for output alignment, never as a predictor. All species statistics, move vocabularies, roster candidates and empirical response tables come from the competition training data. The only supplied lookup table is the Gen 2 type chart.

## 3 Development history

### First baseline and battle state

We started with six team and lead averages in logistic regression. E00 scored 0.5178; scaling gave 0.5164. E01 added last-observed HP, faints, status and active boosts, reaching 0.6671. It became our first submitted approach. HP difference alone scored 0.6519, so the timeline carried far more information than team averages.

E02 added HP trajectories and lead changes but scored 0.6657. E03 added base-stat and type-matchup summaries but scored 0.6665. Neither improved the E01 representation. These results applied to those particular summaries; they did not establish that temporal information or matchups could never help.

### Sacrifices and the first tree ensemble

Error analysis led us to separate Explosion and Self-Destruct from other faint evidence. The resulting E05 model scored 0.6712 with logistic regression. Random forest and default gradient boosting both scored 0.6695, and k-nearest neighbours scored 0.6426 on the same features.

Move counts, observed move effects and surviving-species indicators then favoured boosted trees. The 404-column LightGBM/XGBoost blend scored 0.6875 on the original partition and 0.6841 averaged over seeds 6 to 10, against 0.6690 for E05 on those repeated partitions. We submitted this wider model. A 20-setting LightGBM search did not improve the selected configuration.

### Compact features and team records

The compact model added role counts and retained 176 of 413 columns using permutation importance. It scored 0.6885 on the original partition and 0.6842 over seeds 6 to 10, almost unchanged from the wider model. It was submitted as a smaller alternative.

Recurring team compositions supplied a stronger addition. We calculated smoothed team records inside inner folds. The first logged team model scored 0.6933. We then fixed a tie between possible P2 rosters that depended on Python set order. The corrected model scored 0.6947 on the original partition and 0.6901 over seeds 6 to 10. The submitted team file preceded this correction, so it is not the same predictor as the corrected CV run.

### Win conditions and recording problems

The first win-condition model learned species-pair net HP exchanges and counted attackers with favourable worst matchups against the opposing survivors. It scored 0.6939 on the original partition and 0.6929 over seeds 6 to 10, then became another submission. The chart pooled different move sets, statuses and switching situations. We later revisited the idea with more specific tactical evidence.

We also revised our interpretation of the timeline. P1's extra recorded faints did not prove that P1 was usually behind: ordinary P2 attack knockouts were often unmarked. The state-reconstruction model kept uncertain departures, recovery resources and move records. Grouped validation prevented identical timelines from crossing folds. Fixes covered empty damage counts, opponent leads, fainted roster members, deterministic ties and move lookups that changed their inputs. Affected features were recomputed; unchanged references were reused only when their source and folds were unchanged.

This model scored 0.7020, 0.6995 and 0.7024 on the three grouped partitions, averaging 0.7013 against 0.6932 for the earlier candidate on the same splits. It produced the deterministic state-reconstruction submission. Extra hierarchy features and CatBoost mixtures did not give a consistent further gain.

### Follow-up features and tactical coverage

KO-risk weighting, all compatible P2 rosters and extra move-observation evidence were tested separately and together. Their mean accuracies ranged from 0.70000 to 0.70117, below the 0.70130 state reference, so that round produced no replacement.

Species HP/status features and five-turn forecasts addressed information lost by team totals. The chosen two-group model averaged 0.70360; adding a separate species-move resource block did not help. On an additional partition it scored 0.7071 against 0.7024. This became the species-forecast submission.

The tactical submission revisited win conditions with consistent move-set hypotheses, defensive coverage, recovery, hazards and Explosion trades. It averaged 0.70513 across the three grouped partitions with model seed 0. Nine partition/model-seed comparisons gave a mean gain of 0.00310, with eight positive comparisons. These were sensitivity checks on the same battles.

### Checks before combining the models

A fundamental diagnostic retained the state model: on one existing partition, removing outcome records reduced accuracy from 0.7020 to 0.6948. An inner-selected ordered-observation blend later scored 0.7029 against 0.7024, worsened log loss and failed its screen. We did not promote that standalone procedure, but kept raw summaries and active stints available as alternative representations.

Isolated trials of softer KO treatment, Heal Bell updates and learned zero-power attack pressure averaged 0.70183, 0.70047 and 0.70223. Their joint version scored 0.70140. None passed the replacement guard. Zero-power pressure improved accuracy in all three partitions and made different errors, so it remained a candidate for a combination test.

### The selected combination and submission

Direct feature union averaged 0.70170. Mixing the four core models gave 0.70400; adding raw summaries gave 0.70600, and using active-stint summaries instead gave 0.70640. Learned mixture weights were also tested inside the training folds, but their primary gate failed. We retained fixed weights.

A later check averaged each battle's three held-out probabilities before classification. The raw-summary mixture reached 0.7103, against 0.7092 for the stint mixture and 0.7026 for the state reference. This was a secondary analysis after the fixed-mixture comparison. We then specified a matching test procedure with three routed fold predictions per battle, producing the selected matched-core-summary submission.

Our nine submitted approaches followed this order: E01 state, the wide tree blend, compact features, team records, the first win-condition chart, deterministic state reconstruction, species forecasts, tactical coverage and the matched five-view mixture. The full chronology, filenames, rejected branches and recorded metrics are retained in `docs/development_history.md` and `results/history.json`.

The early experiments used stratified five-fold CV with seed 42, followed by seeds 6 to 10. Later experiments used the grouped partitions described below. Historical scores remain the values recorded at the time; they were not all recomputed with current code. Repeated seeds reuse the same battles, and different evaluation procedures must not be read as one directly comparable score series.

## 4 Validation strategy

We use three stratified grouped five-fold partitions, with seeds 20261026, 20261105 and 193. Each fold trains on 8,000 battles and predicts the other 2,000. Identical timelines cannot cross the training-validation boundary. We keep the same partitions for every component and comparison.

Feature learning follows that boundary. Species and move vocabularies, roster frequencies and move-set examples use only the outer training fold. Features derived from battle outcomes need another split: team and species-move win records use five inner grouped folds. A training battle receives records computed without its own group; validation battles use records fitted on all 8,000 outer training battles.

The same principle applies to auxiliary targets inside the visible timeline. Knockout evidence and empirical attack pressure use two inner folds with seed 347. HP forecasts use two with seed 27183; tactical HP-response tables use two with seed 41071. These auxiliary targets are recorded departures or HP changes, not the final winner. Their training-row estimates still exclude the group being predicted. Outer validation rows never fit any of these estimators.

Each training battle has one held-out prediction in each partition. We average those three probabilities, then classify at 0.5. This differs from averaging the three accuracy scores: the former can correct a decision that varies across partitions, whereas the latter only summarises three separate decisions.

We have compared many ideas on these same battles. New partition seeds do not make the battles an untouched holdout. The reported CV measures this selected method on reused development data and can be optimistic. Reproducing its probabilities checks the implementation; it does not remove selection bias.

## 5 Feature engineering

### Reconstructed state and records

The state view keeps remaining HP, confirmed faints, status counts, active boosts, species still standing, recorded move counts and observed HP changes. Explosion and Self-Destruct observations are separated from other faint evidence. Recovery features attach known Rest, Sleep Talk and other recovery access to the HP of the Pokémon that owns it. We also describe low-HP members, the spread of remaining HP, switch outcomes and stale observations.

Recurring teams supply another source of information. We match the revealed P2 species to compatible rosters found in the training fold, including fully revealed P2 teams. Observed moves reweight those candidates. Team win rates are smoothed as `(wins + 5) / (games + 10)`, and we retain support, uncertainty and the weighted opponent record. A unique P2 match also contributes the opposite outcome to that roster's record. Teams can reflect both composition and player tendencies; we do not have player identities to separate them.

Species-move and move-pair records use a 20-battle prior, with team-specific records shrunk towards the team's rate. All outcome records are cross-fitted. A separate LightGBM classifier learns recorded P1 attack knockouts versus unmarked departures. Its transferred P2 scores describe uncertainty; they do not relabel P2 Pokémon as dead or claim calibrated death probabilities. The state view has 1,085 to 1,089 columns across the outer folds.

### Species state and HP forecasts

Team HP alone loses information about who owns that HP. For the 20 most common P1 species in each training fold, we add HP, sleep HP, paralysis HP, other status HP, observation flags and observation age on both sides, plus differences. These 320 columns distinguish, for example, healthy reserves from recovery users that are asleep or nearly fainted.

We also learn how the observed team-HP proxy changes over five turns. Each training battle supplies transitions 10 to 15, 15 to 20, 20 to 25 and 25 to 30. Two LightGBM regressors predict P1 and P2 changes from the earlier prefix alone. At turn 30, their forecast, implied next HP and error on the already visible 25-to-30 transition supply nine features. No turn after 30 is used. Extending those learned short transitions beyond turn 30 is an approximation, not an observed continuation.

### Tactical coverage and win conditions

Our earlier win-condition idea reduced each species pair to its average net HP exchange. That can mix very different move sets, statuses and switch situations. The current view asks whether the remaining defenders can answer the remaining attackers in the observed state.

We form up to four weighted move-set hypotheses per species from training observations. Each includes the revealed moves and normally respects the four-move limit; copied-move cases are handled separately. A complete revealed set does not receive an extra hypothetical attack. The calculation uses up to three compatible opponent rosters and fixed draws from the move-set hypotheses, keeping the same plausible set across different defenders.

Coverage features combine HP, type effectiveness, attack and defence proxies, active boosts, speed, status and known recovery. They describe unanswered pressure, backup defenders, threats sharing one defender and the effect of losing that defender. Spikes and Rapid Spin observations provide uncertain entry-cost evidence. Explosion trades remove both participants when estimating which answers disappear. We also compare turn-20 and turn-30 resource states while holding the final revealed move information fixed.

These are approximate features. The status penalties, hazard scores, recovery allowances and smooth decision functions are fixed heuristics, not fitted probabilities or exact Gen 2 simulation. The models can learn how useful they are. The tactical view replaces the original threat block and adds 228 columns, giving 1,313 to 1,317 columns in total.

### Learned pressure for zero-power attacks

Some damaging moves have zero recorded base power. Treating them as harmless discards their pressure. We estimate a replacement from conservative observed HP-loss intervals: the active pair stays the same, the recorded attack changes, and several recovery, status, boost-change and near-faint cases are excluded.

Known-power attacks set a median scale. For zero-power moves, we estimate pooled pressure when at least five usable examples exist, then shrink species-specific estimates towards that pool with weight 20. Hidden Power does not use the supplied type field as a revealed type choice. The result is an empirical pressure proxy that may still contain residual HP effects. It replaces the state view's threat values without adding columns.

### Raw observations

The fifth view uses a separate representation. It includes roster and item identities, the final active state, pooled numeric summaries, and each observed species' last state, age and revealed move set. It uses 640 columns and one LightGBM model. Categorical levels are learned from the training fold, so unseen validation or test categories remain unknown.

This view is weaker alone, but preserves observations that our reconstructed features compress or interpret. Its contribution is judged by the combined predictions, not by its standalone accuracy.

## 6 Models and selection

The first four views each average LightGBM and XGBoost with equal weights. Both use 500 trees, learning rate 0.03, row subsampling 0.8, column subsampling 0.5 and seed 0. LightGBM uses 15 leaves and at least 40 samples per leaf; XGBoost uses depth 4 and minimum child weight 5. We keep those settings fixed across partitions. The raw view uses the same LightGBM settings.

We give each view 20% of the fold probability. This means eight core classifiers each receive weight 0.1 and the raw classifier receives weight 0.2. We then average the three partition probabilities. There are 135 winner classifiers fitted in total, but each battle uses only 27 of them. Auxiliary feature estimators are fitted separately.

| View or combination | Mean partition accuracy | Accuracy after averaging OOF probabilities |
|---|---:|---:|
| Reconstructed state | 0.7013 | 0.7026 |
| State with species and forecasts | 0.7036 | 0.7059 |
| State with tactical coverage | 0.7051 | 0.7065 |
| State with learned attack pressure | 0.7022 | 0.7040 |
| Raw observations alone | 0.6548 | 0.6606 |
| Four core views with equal weights | 0.7040 | 0.7043 |
| Five selected views with equal weights | 0.7060 | 0.7103 |

The separate views make different errors, which is why their standalone gains cannot simply be added. The raw model is the weakest alone, yet adding it to the four-core mixture raises averaged OOF accuracy from 0.7043 to 0.7103. Its value here comes from the combined decisions. These comparisons reuse the development partitions, so they describe the selected combination rather than an independent estimate of its gain.

## 7 Results and matching test prediction

The selected mixture scores 0.7071, 0.7047 and 0.7062 in the three partitions. Their mean is 0.7060. Averaging the probabilities before thresholding gives 7,103 correct predictions out of 10,000, or 0.7103. Against the similarly averaged state baseline, it fixes 272 errors and introduces 195, a net gain of 77 battles or 0.77 percentage points. Log loss changes from 0.553175 to 0.553008, a much smaller improvement than accuracy.

The submission uses the same 8,000-row training size and three probabilities per battle. In each partition, a test timeline already found in training uses that group's held-out fold. A new timeline is routed by the integer SHA-256 hash of `matched-core-summary-v1|seed|group`, modulo five. The rule was fixed before test prediction, does not use outcomes and is unchanged by test-row order. Eight test rows match training timeline groups; those groups are excluded from the relevant fitting sets.

We average only the three routed fold mixtures and apply the same 0.5 threshold. Averaging all fifteen fold models or refitting on all 10,000 battles would define a different predictor from the one behind 0.7103. The validation folds are stratified by the training labels, while new test groups use label-free hashing, so matching the training sizes and averaging rule does not make the two sampling procedures identical.

## 8 Reproducibility and remaining limits

The repository has one entry point. `python run.py predict --data data --jobs 2` builds the features from the competition files, fits the fifteen fold ensembles, evaluates OOF predictions and writes `submission_matched_core_summary.csv` under `outputs/predict/`. `python run.py cv --data data` runs validation without loading test data. The Kaggle notebook clones the repository, installs the pinned package versions and invokes the same code.

We keep fold indices, component probabilities, package versions and input/source hashes with each run. Completed folds can be reused only when that signature matches. Stable roster tie-breaking and non-mutating move lookups prevent feature construction from changing with earlier reads. Regression checks cover duplicate grouping, routing, absent moves, faints, prefixes and probability averaging. The released code is also checked against the saved OOF and test probabilities for all fifteen folds and against the submitted CSV hash.

The remaining uncertainty comes from the hidden opponent roster and move choices, stale bench observations, imperfect action timing and all play after turn 30. Our learned response estimates and tactical calculations reduce some of that uncertainty without resolving it. Further comparisons should use the same fixed evaluation procedure and retain the distinction between development CV and performance on unseen battles.
