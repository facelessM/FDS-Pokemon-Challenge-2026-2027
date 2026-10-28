"""Checks for the recording and validation cases that previously caused mistakes."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.base_features import move_effect_features
from src.coverage import alive
from src.move_records import MoveRecords
from src.observations import describe, observation_features
from src.species import snapshots
from src.state import resources
from src.team_context import TeamContext, top_candidate_indices
from src.validation import blend, folds, route


def state(name, hp=1.0, status="nostatus"):
    return dict(
        name=name,
        status=status,
        hp_pct=hp,
        effects=[],
        boosts={
            k: 0 for k in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        },
    )


def battle():
    return dict(
        battle_id=1,
        player_won=True,
        p1_team_details=[dict(name=n, item="leftovers", level=100) for n in "abcdef"],
        p2_lead_details=dict(name="g", level=100),
        battle_timeline=[
            dict(
                turn=i,
                p1_pokemon_state=state("a"),
                p2_pokemon_state=state("g"),
                p1_move_details=None,
                p2_move_details=None,
            )
            for i in range(1, 31)
        ],
    )


class RecordedState(unittest.TestCase):
    def test_empty_damage_has_zero_hits(self):
        f = move_effect_features(battle())
        for side in ["p1", "p2"]:
            for key in ["n_hits", "dmg_mean", "dmg_total", "dmg_max"]:
                self.assertEqual(f[side + "_" + key], 0)

    def test_reading_features_does_not_add_moves(self):
        b = battle()
        view = describe(b)
        before = copy.deepcopy(view)
        observation_features(b, view)
        resources(b, view)
        self.assertEqual(view, before)
        self.assertEqual(view["moves"], {"p1": {}, "p2": {}})

    def test_unobserved_lead_still_constrains_roster(self):
        b = battle()
        for turn in b["battle_timeline"]:
            turn["p2_pokemon_state"]["name"] = "h"
        view = describe(b)
        model = TeamContext()
        model.by = {"g": {("g", "h")}, "h": {("g", "h"), ("h", "i")}}
        self.assertEqual(model.candidates(view), [("g", "h")])

    def test_roster_hypothesis_does_not_revive_a_faint(self):
        view = describe(battle())
        view["states"]["p2"]["g"] = state("g", 0, "fnt")
        self.assertEqual(alive(view, "p2", ["g", "h"]), ["h"])

    def test_prefix_inputs_cannot_see_later_rows(self):
        b = battle()
        changed = copy.deepcopy(b)
        for turn in changed["battle_timeline"][20:]:
            turn["p1_pokemon_state"]["hp_pct"] = 0.1
        original, _, _ = snapshots(b)
        altered, _, _ = snapshots(changed)
        self.assertEqual(original[:3], altered[:3])
        self.assertNotEqual(original[3:], altered[3:])


class Determinism(unittest.TestCase):
    def test_near_tied_rosters_use_canonical_order(self):
        self.assertEqual(
            top_candidate_indices([0.1, 0.1 + 1e-15, 0.8], 2).tolist(), [2, 0]
        )
        self.assertEqual(
            top_candidate_indices([0.1, 0.1001, 0.7999], 2).tolist(), [2, 1]
        )

    def test_move_order_does_not_change_records(self):
        views, labels = [], []
        for i in range(30):
            v = describe(battle())
            v["team"] = ("a", "b", "c", "d", "e" + str(i % 5), "f" + str(i % 5))
            v["moves"] = {
                s: {"a": ["move3", "move1", "move2"], "b": ["move4", "move2"]}
                for s in ["p1", "p2"]
            }
            views.append(v)
            labels.append(int(i % 7 < 3))
        reordered = copy.deepcopy(views)
        for v in reordered:
            for side in ["p1", "p2"]:
                v["moves"][side] = {
                    n: list(reversed(ms))
                    for n, ms in reversed(list(v["moves"][side].items()))
                }
        model = MoveRecords().fit(views, np.array(labels))
        other = MoveRecords().fit(reordered, np.array(labels))
        pd.testing.assert_frame_equal(
            model.transform(views), other.transform(reordered)
        )

    def test_groups_stay_together_and_routing_is_order_invariant(self):
        groups = np.repeat(np.array([f"battle_{i}" for i in range(50)]), 2)
        y = np.repeat(np.arange(50) % 2, 2)
        partitions, assignment = folds(y, groups, 193)
        for train, valid in partitions:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
        test = np.array([groups[0], "new", "new", groups[-1], "another"])
        assigned = route(test, 193, groups, assignment)
        self.assertEqual(assigned[0], assignment[0])
        self.assertEqual(assigned[1], assigned[2])
        np.testing.assert_array_equal(
            assigned[::-1], route(test[::-1], 193, groups, assignment)
        )

    def test_probability_average_is_not_a_vote(self):
        p = np.array([[[0.99] * 5], [[0.49] * 5], [[0.49] * 5]])
        self.assertGreater(blend(p).mean(axis=0)[0], 0.5)
        self.assertLess((blend(p) >= 0.5).mean(axis=0)[0], 0.5)


if __name__ == "__main__":
    unittest.main()
