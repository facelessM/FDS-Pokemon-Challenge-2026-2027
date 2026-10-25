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
