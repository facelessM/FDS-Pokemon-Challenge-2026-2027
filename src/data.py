"""
Loading helpers for the competition files.

Everything that touches the disk lives here, so the rest of the code
can work with plain Python lists and dicts.
"""

import json
from pathlib import Path

import pandas as pd


# Where Kaggle mounts the competition data inside a notebook.
KAGGLE_DATA_DIR = Path("/kaggle/input/competitions/fds-pokemon-battles-prediction-2026")

# When running on our own machines we keep the files in a data/ folder
# next to run.py. The folder is in .gitignore, we never commit the data.
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_data_dir(data_dir=None):
    """Pick the folder that holds train.jsonl.

    If a path is passed explicitly we use it. Otherwise we try Kaggle
    first and then the local data/ folder.
    """
    candidates = [Path(data_dir)] if data_dir else [KAGGLE_DATA_DIR, LOCAL_DATA_DIR]
    for folder in candidates:
        if (folder / "train.jsonl").exists():
            return folder
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find train.jsonl. Looked in: {tried}")


def load_jsonl(path):
    """Read a .jsonl file into a list of dicts (one dict per battle)."""
    battles = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines just in case
                battles.append(json.loads(line))
    return battles


def load_battles(data_dir):
    """Return (train_battles, test_battles) as lists of dicts."""
    data_dir = Path(data_dir)
    train = load_jsonl(data_dir / "train.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")
    return train, test


def load_sample_submission(data_dir):
    """The sample file tells us which battle_ids to predict and in what order."""
    return pd.read_csv(Path(data_dir) / "sample_submission.csv")


def build_species_table(battles):
    """Types and base stats per species, taken from the battles themselves.

    For P2 we only get the details of the lead. Every other P2 Pokemon
    shows up in the timeline by name only. The same species do appear
    with full details on P1 teams in other battles, and a species always
    has the same types and base stats, so we can look them up from there.

    Built from the training battles only. Nothing here comes from outside
    the competition data.
    """
    table = {}
    for battle in battles:
        for poke in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            table[poke["name"]] = {
                "types": [t for t in poke["types"] if t != "notype"],
                "base_total": sum(poke[f"base_{s}"] for s in ("hp", "atk", "def", "spa", "spd", "spe")),
                "base_spe": poke["base_spe"],
            }
    return table


def load_type_chart(data_dir):
    """Read the Gen 2 type chart supplied with the competition.

    The csv only lists matchups that are not neutral, so anything
    missing from the file counts as x1.0. We return a dict keyed by
    (attacking_type, defending_type).

    We search for the file with a glob because the exact name on Kaggle
    may differ a bit from the one we downloaded.
    """
    matches = sorted(Path(data_dir).glob("*type_chart*.csv"))
    if not matches:
        raise FileNotFoundError(f"No type chart csv found in {data_dir}")
    chart = pd.read_csv(matches[0])
    return {
        (row.attacking_type, row.defending_type): float(row.multiplier)
        for row in chart.itertuples(index=False)
    }
