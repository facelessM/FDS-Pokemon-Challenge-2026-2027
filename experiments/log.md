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
- **Leaderboard:** to be submitted.
- **Takeaway:** a static picture of turn 30 already moves us from a coin flip to about
  two battles out of three. Some features are redundant (`n_alive` is just
  `6 - n_fainted`). That does not hurt a regularised logistic regression, but we will
  prune it in the feature selection step (E07).
- **Decision:** keep `state_t30`. Next is E02: this snapshot says nothing about *how*
  the battle got there, so we look at the trend over the 30 turns.
