# A wild Gengar Appears!

**Pokémon battle outcome prediction** · Foundations of Data Science, Sapienza Università di Roma, 2026/2027

Given Player 1's full team, Player 2's lead Pokémon and the first 30 turns of a
Gen 2 OU battle, we predict whether Player 1 goes on to win. Battles last between
30 and 80 turns, so the model only ever sees the opening part of the game.

## Team

| Name | Student ID | Email |
|---|---|---|
| Özgür Cihan Karahan | 2233702 | karahan.2233702@studenti.uniroma1.it |
| Gabriele Fiacconi | 2060444 | fiacconi.2060444@studenti.uniroma1.it |
| Musa (TBD) | TBD | TBD |

## Results so far

Accuracy from stratified 5-fold cross validation on the 10,000 training battles.
Each row links to the commit that produced it and is explained in
[`experiments/log.md`](experiments/log.md).

| Exp | What changed | CV accuracy | Public LB |
|---|---|---|---|
| E00 | Starter notebook features + logistic regression | 0.5178 ± 0.0080 | |
| E01 | + battle state at turn 30 (faints, team HP, statuses, active Pokémon) | 0.6671 ± 0.0046 | |

## Repository layout

```
run.py                 entry point: features -> cross validation -> refit -> submission
src/
  data.py              loads the battles, the sample submission and the type chart
  features.py          feature groups, one function each, plus a registry
  models.py            model definitions (classical ML only)
  validation.py        stratified 5-fold CV with a fixed seed
  experiments.py       one config per experiment: which feature groups, which model
experiments/
  log.md               what we tried, why, what happened, what we decided
  results.csv          raw CV numbers, appended by run.py with the commit hash
notebooks/
  kaggle.ipynb         clones this repo on Kaggle and builds the final submissions
```

## Quick start

Put the competition files in a `data/` folder at the root of the repo. The folder is
ignored by git, we never commit the data.

```
data/train.jsonl
data/test.jsonl
data/sample_submission.csv
data/type_chart_gen2.csv
```

Then run any experiment by name:

```bash
pip install -r requirements.txt
python run.py --exp e00_starter
```

You get the CV score per fold, a new line in `experiments/results.csv` and the
submission file in `submissions/e00_starter.csv`.

## Running on Kaggle

Open `notebooks/kaggle.ipynb` in a notebook attached to the competition, switch on
Internet access and run all cells. While the repo is private the notebook needs a
GitHub token saved in Kaggle Secrets as `GITHUB_TOKEN`. After the competition the
repo goes public and the token is no longer needed.

The notebook does not contain any modelling code of its own. It clones the repo and
calls `run_experiment`, so the submissions come from exactly the code in this repo.

## How we work

- **One experiment, one change.** Each idea gets its own entry in
  `src/experiments.py`. Old entries are never edited, so every number in the log can
  be reproduced with `python run.py --exp <name>`.
- **Local CV decides, the leaderboard only confirms.** All folds are fixed with
  `random_state=42`. A change is kept only if the gain is clearly above the
  fold-to-fold noise (rule in `experiments/log.md`). We submit at most once or twice a
  day, as a sanity check.
- **Every commit says what changed and what it did to the score.** The log entry
  explains the reasoning, the commit message gives the short version.

## Competition rules we follow

- Classical machine learning only, no neural networks and no pretrained models.
- Only the competition data and the supplied Gen 2 type chart. No external databases,
  replay datasets, usage statistics or hand-made tables.
- The test set is never used to choose features or tune parameters.
