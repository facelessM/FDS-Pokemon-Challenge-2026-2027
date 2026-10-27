"""
Main entry point.

    python run.py --exp e00_starter

What it does, in order:
  1. loads train and test battles
  2. builds the feature table for the chosen experiment
  3. runs 5-fold cross validation and prints the scores
  4. appends the result to experiments/results.csv
  5. refits the model on the full training set
  6. writes submissions/<exp>.csv in the format Kaggle expects

The Kaggle notebook calls this same script, so what we see locally and
what we submit always come from the same code.
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import (build_matchup_chart, build_move_vocab, build_species_table,
                      build_team_index, build_top_species, find_data_dir,
                      load_battles, load_sample_submission, load_type_chart)
from src.experiments import EXPERIMENTS
from src.features import build_features, get_target
from src.models import get_model
from src.validation import SEED, cross_validate


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "experiments" / "results.csv"


def current_commit():
    """Short git hash of the code we are running, so each result in
    results.csv can be traced back to a commit. Returns 'nogit' if git
    is not available (it usually is, also on Kaggle)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "nogit"


def log_result(exp_name, cv, n_features):
    """Append one line to experiments/results.csv.

    This file is the raw record. The written explanation of each
    experiment (why we tried it, what we concluded) goes in log.md.
    """
    is_new = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["date", "commit", "experiment", "n_features",
                             "cv_mean", "cv_std", "fold_scores"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            current_commit(),
            exp_name,
            n_features,
            f"{cv['mean']:.4f}",
            f"{cv['std']:.4f}",
            " ".join(f"{s:.4f}" for s in cv["fold_scores"]),
        ])


def make_submission(test_ids, test_pred, sample, out_path):
    """Write predictions using sample_submission.csv as the template.

    We map by battle_id rather than trusting the row order, and fail
    loudly if any id is missing, because a broken file wastes one of our
    3 daily submissions.
    """
    if list(sample.columns) != ["battle_id", "player_won"]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")

    pred_by_id = dict(zip(test_ids, test_pred))
    sub = sample.copy()
    sub["player_won"] = sub["battle_id"].map(pred_by_id)
    if sub["player_won"].isna().any():
        raise ValueError("Some battle_ids in the sample file have no prediction.")
    sub["player_won"] = sub["player_won"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub


def run_experiment(exp_name, data_dir=None, out_dir=None, log=True, model=None):
    if exp_name not in EXPERIMENTS:
        sys.exit(f"Unknown experiment '{exp_name}'. Available: {', '.join(EXPERIMENTS)}")
    config = dict(EXPERIMENTS[exp_name])

    # Same features, different model (used for the gradient boosting check).
    # The run is logged and saved under a name that shows the swap.
    if model and model != config["model"]:
        config["model"] = model
        exp_name = f"{exp_name}[{model}]"

    # 1. Data
    data_dir = find_data_dir(data_dir)
    print(f"Data folder: {data_dir}")
    train, test = load_battles(data_dir)
    print(f"Loaded {len(train)} train and {len(test)} test battles")

    # Shared resources that some feature groups need. The species table is
    # built from the training battles only and holds no labels.
    context = {
        "type_chart": load_type_chart(data_dir),
        "species": build_species_table(train),
        "move_vocab": build_move_vocab(train),
        "top_species": build_top_species(train),
        "teams": build_team_index(train),
        "matchups": build_matchup_chart(train),
    }

    # 2. Features. We build train and test with the same function and
    # then force the test columns to match train, so a feature that only
    # shows up in one of the two sets cannot shift the columns.
    train_df = build_features(train, config["features"], context)
    test_df = build_features(test, config["features"], context)
    feature_cols = [c for c in train_df.columns if c != "battle_id"]
    # Some experiments keep only a chosen subset of the columns.
    if "columns" in config:
        keep = Path(__file__).resolve().parent.joinpath(config["columns"]).read_text().split()
        feature_cols = [c for c in feature_cols if c in keep]
    test_df = test_df.reindex(columns=["battle_id"] + feature_cols, fill_value=0)

    X_train, y_train = train_df[feature_cols], get_target(train)
    X_test = test_df[feature_cols]
    print(f"Experiment {exp_name}: {len(feature_cols)} features, model={config['model']}")

    # 3. Cross validation (timed, since run time matters when comparing models)
    start = time.time()
    cv = cross_validate(config["model"], X_train, y_train)
    folds = ", ".join(f"{s:.4f}" for s in cv["fold_scores"])
    print(f"CV accuracy: {cv['mean']:.4f} +/- {cv['std']:.4f}  (folds: {folds})  [{time.time() - start:.0f}s]")

    # 4. Record it
    if log:
        log_result(exp_name, cv, len(feature_cols))

    # 5. Refit on all the training data
    model = get_model(config["model"], SEED)
    model.fit(X_train, y_train)
    test_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    # 6. Submission file
    out_dir = Path(out_dir) if out_dir else ROOT / "submissions"
    out_path = out_dir / f"{exp_name}.csv"
    sub = make_submission(test_df["battle_id"].values, test_pred,
                          load_sample_submission(data_dir), out_path)
    print(f"Wrote {out_path} ({len(sub)} rows, predicted win rate {np.mean(test_pred):.3f})")

    return cv, sub


def main():
    parser = argparse.ArgumentParser(description="Pokemon battle prediction")
    parser.add_argument("--exp", required=True, help="experiment name from src/experiments.py")
    parser.add_argument("--data-dir", default=None, help="folder with train.jsonl (auto-detected if omitted)")
    parser.add_argument("--out-dir", default=None, help="where to write the submission csv")
    parser.add_argument("--no-log", action="store_true", help="do not append to experiments/results.csv")
    parser.add_argument("--model", default=None, help="run the experiment's features with another model")
    args = parser.parse_args()

    run_experiment(args.exp, args.data_dir, args.out_dir, log=not args.no_log, model=args.model)


if __name__ == "__main__":
    main()
