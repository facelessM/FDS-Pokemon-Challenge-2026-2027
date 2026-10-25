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
