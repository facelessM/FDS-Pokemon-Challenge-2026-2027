# Experiment log

One entry per experiment. Raw numbers are in `results.csv` (written by `run.py`),
this file explains why we tried something and what we concluded.

**Validation setup (same for every entry):** stratified 5-fold CV on the 10,000
training battles, `random_state=42`, accuracy at a 0.5 threshold.

**When do we call something an improvement?** With 2,000 battles per fold, one fold
score moves by about ±1.1 points just by chance (binomial standard error at 50%), and
the fold std we see is around 0.8 points. So we treat gains under about 0.3 points of
mean CV accuracy as noise, unless they show up on every fold.

---

## Data notes (first look, 25 Oct)

- 10,000 train and 5,000 test battles. Classes are perfectly balanced (50.0% wins),
  so plain accuracy is a fair metric and the trivial baseline is 50%.
- Every battle has exactly 30 timeline entries, turns 1 to 30.
- Player 1 always has 6 Pokémon. 237 different species appear on P1 teams. The most
  common are Snorlax, Cloyster, Zapdos, Raikou and Tyranitar.
- Almost everything is level 100 (59,640 of 60,000 slots). A handful of low level
  Pokémon exist, probably joke or troll teams. Worth a flag feature later.
- Items: Leftovers on about 95% of slots, then Thick Club, Miracle Berry, Mint Berry.
- About 20% of turns have no move for a side (`move_details` is null). That should
  mean a switch, or a turn where the Pokémon could not act (sleep, full paralysis,
  recharge). We need to check which one before using it.
- Status values: `nostatus`, `slp`, `par`, `fnt`, `tox`, `psn`, `frz`, `brn`.
  Fainted Pokémon show up as `fnt` with `hp_pct == 0`, so we can count KOs per side.
- Player 2's team is hidden. We only learn about P2 Pokémon when they appear as the
  active Pokémon in the timeline. This asymmetry needs care in the features
  (for P2, "unseen" is not the same as "full HP").

---

## E00: starter baseline

- **Commit:** `45d623a`
- **Hypothesis:** none, this is the reference point. Same 6 features as the
  starter notebook (P1 team mean HP/Spe/Atk/Def, P2 lead HP/Spe) and the same
  logistic regression.
- **Result:** CV **0.5178 ± 0.0080** (folds 0.5145 / 0.5190 / 0.5325 / 0.5090 / 0.5140)
- **Leaderboard:** not submitted yet
- **Takeaway:** barely better than a coin flip. Team averages alone say almost
  nothing about who wins, which makes sense: in OU most teams use similar Pokémon
  with similar stats, so the averages hardly vary between battles. The real signal
  has to come from the timeline (who is ahead by turn 30) and from matchups.
- **Decision:** keep as the baseline. Next step is E01/E02, starting with the state
  of the battle at turn 30 since that is where we expect most of the signal.

---

## E01: battle state at turn 30

- **Commit:** `e2e6a82`
- **Hypothesis:** the clearest signal in a 30-turn snapshot should be who is ahead
  at the end of it: how many Pokémon each side has lost, how much HP the teams have
  left, and who is carrying status conditions.
- **What we built** (feature group `state_t30`, 33 new features):
  - For each side we keep the *last seen* state of every Pokémon that appeared in the
    timeline. A benched Pokémon keeps the HP and status it had when it left the field,
    so its last appearance is still accurate at turn 30.
  - Per side: fainted, alive and seen counts; total remaining HP over the whole team;
    number of living Pokémon with each status; HP, status and stat boosts of the
    active Pokémon.
  - P1 minus P2 differences for alive count, team HP, statused count, active HP and
    active boosts.
- **Design choice: Pokémon we have not seen.** P2's team is hidden, so after 30 turns
  we usually know only 4 to 6 of its Pokémon. We count every unseen Pokémon as being
  at full HP, for both sides. This is not a guess: in Gen 2 a Pokémon cannot take
  damage before it switches in (Spikes only hurt on entry). We assume P2 also has six
  Pokémon, as every P1 team does.
- **Data observation:** P1 loses far more Pokémon than P2 in the first 30 turns
  (1.9 vs 0.5 on average) and yet wins exactly half the battles. So the raw faint
  counts are not symmetric between the two sides, and the model has to learn the
  offset rather than just compare the counts.
- **Control run:** `e00b_starter_scaled` (starter features with scaling) gives
  0.5164 ± 0.0063, the same as E00 within noise. The scaler on its own adds nothing,
  so the whole E01 gain comes from the new features.
- **Result:** CV **0.6671 ± 0.0046** (folds 0.6625 / 0.6725 / 0.6610 / 0.6710 / 0.6685),
  **+14.9 points** over E00 and better on every fold.
- **Side checks** (same folds, not separate experiments):
  - `diff_hp_total` alone reaches 0.6519. Remaining team HP carries most of the
    signal on its own.
  - Faint counts alone reach only 0.6020. Losing a Pokémon matters, but how much HP
    the rest of the team still has tells more.
  - Dropping the six starter features: 0.6670, no change. They can go once we clean
    up the feature set.
  - Not counting unseen P2 Pokémon as full HP: 0.6670, no change either. A linear
    model can make up for it through `p2_n_seen`. We keep the full-HP version because
    it is the one that matches the game mechanics, and it may matter more for tree
    models later.
  - Largest standardised coefficients: `diff_hp_total` (+0.51), `p2_hp_total` (-0.44),
    `diff_n_alive` (+0.16), `p1_n_fainted` (-0.14), `diff_active_boosts` (+0.12).
- **Leaderboard:** public score **0.67866** (first submission, 25 Oct). That is 1.2 points
  above CV. The public board scores only part of the 5,000 test battles, so its own
  standard error is around one point, and a gap of this size is what we would expect
  from chance. CV and leaderboard agree, which is the first sign that our local
  validation can be trusted.
- **Takeaway:** a static picture of turn 30 already moves us from a coin flip to about
  two battles out of three. Some features are redundant (`n_alive` is just
  `6 - n_fainted`). That does not hurt a regularised logistic regression, but we will
  prune it in the feature selection step (E07).
- **Decision:** keep `state_t30`. Next is E02: this snapshot says nothing about *how*
  the battle got there, so we look at the trend over the 30 turns.

---

## E02: battle trajectory over the 30 turns

- **Commit:** `25347dc`
- **Hypothesis:** two battles can look the same at turn 30 and still be heading in
  opposite directions. In one P1 has led from the start, in the other P1 was far behind
  and has just caught up. If momentum carries into the hidden turns, the path to turn 30
  should add information on top of the E01 snapshot.
- **What we built** (feature group `trajectory`, 19 new features): we rebuild each
  team's total HP and faint count after every turn, using the same last-seen logic as
  E01, and summarise the HP difference between the teams:
  - its value at turns 5, 10, 15, 20 and 25 (turn 30 is already in E01)
  - mean, minimum, maximum and slope over the 30 turns
  - number of turns with P1 ahead, and number of lead changes
  - change over the last 5 and last 10 turns, per side and as a difference
  - turn of the first faint and number of faints in the last 10 turns, per side
- **Model:** unchanged (scaled logistic regression), so the comparison with E01 is
  about the features only.
- **Result:** CV **0.6657 ± 0.0061** (folds 0.6570 / 0.6720 / 0.6600 / 0.6715 / 0.6680).
  That is 0.14 points *below* E01, inside the noise, and lower on four of five folds.
  The trajectory adds nothing.
- **Why not?** We tested three explanations (`analysis/e02_checks.py`, same folds):
  - **H1, the information is already in E01.** The trajectory features alone reach
    0.6534, almost as much as E01, and the strongest ones follow the turn-30 HP
    difference closely: correlation 0.90 for the slope, 0.89 for the minimum, 0.87 for
    the value at turn 25. They mostly restate where the battle ended up.
    **Supported.**
  - **H2, the signal is non-linear.** Momentum could matter only in combination with
    other features ("ahead, but losing ground"), which a linear model cannot express.
    A gradient boosting model with default settings scores 0.6707 on the E01 features
    and 0.6700 with the trajectory added. No gain there either. **Rejected.**
  - **H3, a few features help but drown in the rest.** Adding each trajectory feature
    to E01 on its own moves CV by at most +0.16 points (HP difference at turn 10),
    below our noise threshold. **Rejected.**
- **Takeaway:** in this data, where the battle stands at turn 30 already sums up how
  it got there. Knowing the path does not help predict the hidden turns. Our hypothesis
  was wrong, and we would rather report that than keep features that only add noise.
  A side result from H2: gradient boosting on the E01 features (0.6707) is slightly
  ahead of logistic regression (0.6671). We will look at this properly in E06.
- **Decision:** do **not** carry the `trajectory` group forward. It stays in the code
  so E02 can be reproduced, but E03 builds on the E01 feature set.

---

## E03: team strength and type matchups

- **Commit:** `e0b9560`
- **Hypothesis:** the starter features failed because they described P1's team on its
  own, and most OU teams look alike. What should matter is how the two teams compare:
  who has the type advantage, and whether P2's revealed Pokémon are stronger than P1's.
- **A data problem first.** For P2 we only get the details of the lead. Every other P2
  Pokémon appears in the timeline by name only. The same species do appear with full
  details on P1 teams in other battles, and a species always has the same types and base
  stats (239 species, no conflicting entries). So we build a species table from the
  training battles and look P2's Pokémon up there. It covers 99.98% of P2 timeline rows;
  the few missing species (Natu, Unown, Remoraid...) are skipped. This uses only the
  competition data, so it stays within the rules.
- **What we built** (feature group `matchup`, 16 new features):
  - mean base stat total of P1's team and of P2's revealed Pokémon, and the difference
  - fastest Pokémon on each side
  - team build: number of distinct types, Pokémon below level 100, Leftovers count
  - type matchups, using the supplied chart and each Pokémon's own types (the best
    STAB multiplier, since we do not know the movesets): P1 team against P2's lead,
    team against team over all revealed pairs, and the two active Pokémon at turn 30
- **Model:** unchanged (scaled logistic regression), built on the E01 feature set.
- **Result:** CV **0.6665 ± 0.0058** (folds 0.6605 / 0.6745 / 0.6615 / 0.6725 / 0.6635),
  0.06 points below E01. No gain.
- **Why not?** (`analysis/e03_checks.py`, same folds)
  - **The signal is weak, not redundant.** On their own the matchup features reach only
    0.5267, one point above the scaled starter control (0.5164). Their correlation with
    the turn-30 HP difference is between -0.07 and +0.12, so this is not E02 again: the
    information is not hidden inside the turn-30 state, there is simply little of it.
  - The best single feature, the type edge between the two active Pokémon, adds
    0.23 points on its own, below our threshold.
  - Gradient boosting with default settings: 0.6714 with the matchup features against
    0.6707 without. No gain.
- **Takeaway:** what the teams look like on paper says little about who wins. Two
  likely reasons: OU teams are built to cover their weaknesses, and a type matchup based
  on a Pokémon's own types misses the coverage moves it actually carries. What happens
  in the battle matters far more than the matchup before it.
- **Decision:** `matchup` does not join the main feature set. Both groups that failed
  so far (`trajectory`, `matchup`) stay in the code and will be checked once more in the
  ablation step (E07) with the final tuned model, in case a non-linear model makes use
  of a weak signal that a linear one cannot. E04 builds on the E01 feature set.

---

## Plan update after E03

Two feature groups in a row (trajectory, matchup) brought nothing. Both were
reasonable ideas, but they came from intuition about the game rather than from the
data. Before adding more features we change how we pick them.

- **Error analysis comes next (E04).** We already keep the out-of-fold predictions of
  every run. We look at where E01 goes wrong (close battles? battles P1 leads at turn
  30 and still loses?) and let that decide what to build next.
- **We go deeper where the signal is.** E01 counts every Pokémon's HP the same, so a
  half-health Snorlax weighs as much as a half-health Pikachu. Strength of the
  Pokémon still standing, and lasting effects (boosts, Substitute, Rest sleep versus
  normal sleep), are the natural next step for the turn-30 state.
- **Every experiment now reports two scores:** the scaled logistic regression as the
  main score, plus a default gradient boosting model as a check, so a non-linear signal
  cannot slip past us.

| # | Step |
|---|---|
| E04 | Error analysis of E01 (out-of-fold predictions) |
| E05 | Richer turn-30 state: strength-weighted HP, lasting effects, sleep type |
| E06 | Moves and player behaviour |
| E07 | Species identity with in-fold target encoding |
| E08 | Model comparison |
| E09 | Feature selection and group ablation, including `trajectory` and `matchup` |
| E10 | Hyperparameter tuning inside CV |
| E11 | Ensembling |
| E12 | Robustness over several CV seeds, choice of the three final submissions |

---

## E04: error analysis of E01

- **Script:** `analysis/e04_error_analysis.py` (analysis only, no new feature group)
- **Question:** where does E01 go wrong, and what information is it missing?
- **Confidence.** The model is well calibrated: accuracy rises from 53% on battles it
  calls 50/50 to 87% on the ones it is most sure about. But 34% of battles sit within
  10 points of 50/50, and there it is barely better than a coin flip. Even when it gives
  P1 more than a 70% chance, P1 still loses 22% of those battles.
- **Residuals.** We correlated candidate quantities that E01 does not use with the
  residual (outcome minus predicted probability). Candidates for a richer turn-30
  state (HP weighted by species strength, healthy and critical counts, Rest sleep versus
  other sleep, volatile effects) all stay below |r| = 0.025. Player behaviour counts
  (attacks, status moves, switches) correlate the most, up to |r| = 0.057, and a few
  species (Snorlax, Machamp, Jolteon...) still alive at turn 30 reach 0.03 to 0.045.
- **Do they turn into accuracy?** Added to E01, on the same folds:

  | Added to E01 | LR | GB |
  |---|---|---|
  | nothing | 0.6671 | 0.6707 |
  | behaviour counts | 0.6650 | 0.6721 |
  | richer turn-30 state | 0.6661 | 0.6650 |
  | species still alive (40 most common, both sides) | 0.6682 | 0.6731 |

  None passes our threshold. The richer state we had planned for the next step would
  not have helped, which is exactly what this analysis was for.
- **Faint counts at turn 30.** In none of the 10,000 battles has P2 lost more than one
  Pokémon more than P1, and P1 loses 1.9 Pokémon on average against 0.5 for P2, yet
  wins half of the battles. We first read this as "P1 is almost always behind, the task
  is predicting comebacks". That reading was wrong: P2's knockouts are mostly not
  recorded as faints (see *Data check: hidden knockouts* below). Both sides actually
  lose about the same number of Pokémon.
- **Can we trust the faint counts?** Every use of Explosion or Self-Destruct is marked
  as fainted, and the low-HP Pokémon that leave and never come back do not explain the
  residual (|r| <= 0.028). This only shows the model does not miss them, not that the
  counts are complete.
- **One thing the checks did show:** P1 uses Explosion and Self-Destruct far more than
  P2. Part of P1's extra faints are sacrifices, not knockouts taken, and E01 counts both
  the same way. That is a fix to the turn-30 state with a clear reason behind it, so it
  becomes E05.
- **Decision:** the feature side is close to what this data allows. After E05 we move
  to the model: comparison, the weak groups bundled with the best model, tuning and
  ensembling. Updated plan:

  | # | Step |
  |---|---|
  | E05 | Separate sacrifices from knockouts taken |
  | E06 | Model comparison |
  | E07 | Weak groups (trajectory, matchup, behaviour, species alive) with the chosen model |
  | E08 | Hyperparameter tuning inside CV |
  | E09 | Ensembling |
  | E10 | Robustness over several CV seeds, choice of the three final submissions |

---

## E05: sacrifices are not knockouts

- **Commit:** `6a6b8b1`
- **Hypothesis:** a Pokémon that uses Explosion or Self-Destruct chooses to go down,
  usually after a heavy hit on the opponent. That is a trade, not a loss, and E01
  counts it the same as being knocked out. Since P1 sacrifices far more often than P2
  (E04), splitting the two should give the model a cleaner picture of who is behind.
- **What we built** (feature group `sacrifice`, 6 features): for each side the number
  of sacrifices and the number of knockouts taken (fainted minus sacrifices), and the
  P1 minus P2 difference of both.
- **Result:** CV **0.6712 ± 0.0061** (folds 0.6630 / 0.6815 / 0.6680 / 0.6705 / 0.6730),
  **+0.41 points** over E01, better on four of five folds.
- **Gradient boosting check:** 0.6695 against 0.6707 for E01. No gain there: a tree
  model can already tell the two kinds of faint apart from the counts it has, a linear
  model cannot.
- **Decision:** keep `sacrifice`. The gain is above our threshold for the main model and
  has a clear reason behind it. The E05 feature set (starter + state_t30 + sacrifice) is
  the base for the model comparison in E06.

---

## E06: model comparison

- **Commit:** `53b5bc0` (branch `e06-model-comparison`)
- **Question:** with the feature set fixed at E05, which classical model family works
  best? Each one rests on a different assumption about the data, and all of them run
  with standard settings so the comparison is fair. Tuning comes in E08.
- **Models:**
  - logistic regression on standardised inputs: the effects are linear
  - k-nearest neighbours (k = 89, the square root of a training fold): similar battles
    end the same way
  - random forest (300 trees): interactions matter, and averaging many deep trees
    keeps the variance down
  - gradient boosting (scikit-learn defaults): interactions matter, and each tree fixes
    the errors of the previous ones
- **Result** (same folds, `python run.py --exp e05_sacrifice --model <name>`):

  | Model | CV accuracy | CV time |
  |---|---|---|
  | Logistic regression | **0.6712 ± 0.0061** | < 1 s |
  | Random forest | 0.6695 ± 0.0040 | 8 s |
  | Gradient boosting | 0.6695 ± 0.0038 | 1 s |
  | k-NN | 0.6426 ± 0.0102 | < 1 s |

- **Reading it:** logistic regression, random forest and gradient boosting are within
  0.17 points of each other, well inside the noise. k-NN is clearly behind: with 45
  features of very different importance, a plain distance gives too much weight to the
  ones that do not matter. That the untuned tree models do not beat a linear model fits
  what we saw so far: the signal is mostly the difference in Pokémon and HP left at turn
  30, and that is close to linear.
- **Do the models make the same mistakes?** (`analysis/e06_checks.py`) The three main
  models' predicted probabilities correlate at 0.88 to 0.91, and they disagree on only 12
  to 15% of battles. Where they disagree, each one is right about half the time. 23% of
  battles are missed by all three. So combining them (E09) can only bring a small gain,
  and we should not expect more.
- **Decision:** carry two models forward. Logistic regression is the best and simplest,
  and stays the main model. Gradient boosting has the most to gain from tuning, since its
  defaults are generic. Random forest behaves almost like gradient boosting (correlation
  0.91) at eight times the cost, and k-NN is clearly behind, so both are dropped.

---

## Final model

- **Commit:** `072e8dc` (branch `final-model`), `python run.py --exp final_blend`
- **Features (404):** E05 plus three groups. *Move usage*: how often each side used
  each move. *Move effects*: damage per attack, attacks with no effect, HP healed,
  turns lost to status. *Species still standing*: for the 60 most common species,
  whether each side still has it at turn 30.
- **Model:** LightGBM + XGBoost, win probabilities averaged. These features are wide
  and sparse, which suits trees; logistic regression did worse on them.
- **Result:** CV 0.6875 ± 0.0058. Public leaderboard **0.6893**. The code reproduces
  the submitted file exactly (5,000 of 5,000 predictions).
- **Honest estimate:** seed 42 was also used while developing these features, so 0.6875
  is slightly optimistic. On CV seeds 6 to 10, never used during development
  (`analysis/final_checks.py`), the final model scores **0.6841 ± 0.0017** against
  0.6690 for E05: a gain of 1.5 points, on every seed.

---

## Model checks behind the final choice

`analysis/model_checks.py` (seed 42) and `analysis/final_checks.py` (seeds 6 to 10).

| Final features, model | CV seed 42 | CV seeds 6-10 |
|---|---|---|
| *LightGBM on E05 features (reference)* | *0.6724 ± 0.0043* | |
| Logistic regression | 0.6713 ± 0.0136 | |
| Gradient boosting, untuned | 0.6739 ± 0.0063 | |
| LightGBM | 0.6878 ± 0.0044 | 0.6820 ± 0.0025 |
| XGBoost | 0.6851 ± 0.0047 | |
| LightGBM + XGBoost | 0.6875 ± 0.0058 | **0.6841 ± 0.0017** |

- With the model fixed, the new feature groups add 1.5 points (LightGBM: 0.6724 on E05
  features, 0.6878 on the final ones), so the gain comes from the features.
- Logistic regression gets nothing out of the 359 new columns (0.6713, the E05 level,
  with twice the fold spread): the move counts are wide and sparse, and their effect
  depends on the battle state, which a linear model cannot combine.
- On seed 42 the blend and LightGBM alone are level. On the untouched seeds the blend is
  ahead on four of five and varies less, which is why we keep it.
- A random search over 20 LightGBM settings did not beat ours (best 0.6843, median
  0.6804). The mild settings (small trees, slow learning rate, subsampling) are enough.

---

## Gen 2 mechanics, shown directly

`analysis/mechanics_tables.py`. Writing the mechanics as explicit features (Spikes that
cannot be cleared, Explosion trade quality, setup threats left unanswered, how well
attacks matched their targets) did not change the CV score: the final model already
picks them up through HP, move usage and move effects. On their own they are strong:

- Spikes stuck on P1's side at turn 30: P1 wins 40.5%. Stuck on P2's side: 58.8%.
- P1's win rate rises from 42.2% to 56.4% across quintiles of attack effectiveness
  (P1 minus P2).
- The final model is least accurate in long, stall-heavy battles: 71.9% with no Rest
  user, 59.4% with four or more. Those battles are decided in turns we cannot see.

---

## Data check: hidden knockouts

`analysis/data_checks.py`. We looked at what the timeline shows right after a Pokémon
below 30% HP takes an attack:

| Side | Fainted | Gone, never seen again | Fainted + never seen again |
|---|---|---|---|
| P1 | 39.8% | 16.1% | 5,572 |
| P2 | 3.0% | 50.4% | 5,745 |

- When P1 knocks out a P2 Pokémon with an attack, the timeline usually does not mark it
  as fainted: it leaves the field and never comes back. 98% of the recorded P2 faints
  happen on the same row as a P1 faint, i.e. Explosion and Self-Destruct trades.
- Counted together, both sides lose about the same number of Pokémon. The E04 reading
  ("P1 is almost always behind") was an effect of this recording, not of the game.
- Adding the hidden knockouts (0.54 per battle) to the final features does not change
  the score (LightGBM, seeds 1 to 5: 0.6818 against 0.6827). The missing faints were
  Pokémon with little HP left, and the model already sees how many P2 Pokémon appeared
  and how much damage each side dealt.
- HP drops line up with the attack on the same row for both sides, so the damage
  features are measured correctly.

---

## Limits of the data

`analysis/limits.py`, out-of-fold predictions of the final model (seed 42).

| P1 at turn 30 | Battles | P1 wins | Model accuracy |
|---|---|---|---|
| 3+ Pokémon down | 762 | 15.0% | 85.3% |
| 2 down | 1,328 | 32.2% | 70.3% |
| 1 down | 2,752 | 44.7% | 64.8% |
| level | 3,859 | 57.9% | 64.7% |
| 1 up | 1,072 | 75.3% | 76.9% |
| 2+ up | 227 | 82.4% | 82.8% |

- Comebacks are common: one Pokémon down, P1 still wins 45% of the time; two down,
  32%. With up to 50 turns left and recovery moves everywhere, a deficit at turn 30 is
  a handicap, not a verdict.
- In every group the model is at least as accurate as always guessing the more likely
  outcome, so it already uses the material lead fully.
- The errors sit in the close battles: 30% of battles are within 10 points of 50/50
  and the model gets 55.8% of them right, against 88.1% for the 21% it is most sure
  about. These battles are most likely decided by what we cannot see: P2's unrevealed
  Pokémon and moves, and the turns after 30.

---

## Compact model

- **Branch:** `compact-model`. `python analysis/compact_selection.py` chooses the
  columns, `python run.py --exp compact_blend` scores the model and writes its
  submission.
- **Idea:** the final model has 404 columns, most of them sparse move counts. Gen 2 OU
  is played around a few roles, so we count, for each side, the Pokémon that used
  Spikes, Rapid Spin and Explosion (feature group `roles`), and then keep only the
  columns the model actually relies on.
- **Selection:** LightGBM fitted on 80% of the training battles; on the other 20% each
  column is shuffled in turn (permutation importance, 5 repeats). Columns whose
  shuffling makes the predictions worse are kept: 176 of 413, of which 76 of the 205
  move columns, 48 of the 120 species columns and 5 of the 9 role columns. The HP
  difference at turn 30 matters far more than any other column (about five times the
  next one), followed by P2's remaining HP, the difference in statused Pokémon, Spikes
  used by each side, damage per hit, P1 still having Snorlax and P2's active boosts.
- **Result:** CV seeds 6 to 10 **0.6842 ± 0.0016**, against 0.6841 ± 0.0017 for the
  final model. On the usual folds 0.6885 ± 0.0063, slightly optimistic since the
  selection saw the same battles. Its test predictions agree with our submitted file on
  95% of battles.
- **Decision:** the same accuracy with less than half the columns. We keep it as our
  second final submission.
