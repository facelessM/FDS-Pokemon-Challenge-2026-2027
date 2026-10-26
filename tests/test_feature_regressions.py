import unittest
from copy import deepcopy

from src.data import build_team_index
from src.features import move_effect_features, team_id_features, win_condition_features


def state(name, hp=1.0, status="nostatus"):
    return {"name": name, "hp_pct": hp, "status": status,
            "effects": ["noeffect"],
            "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe")}}


def row(a, b):
    return {"turn": 1, "p1_pokemon_state": a, "p2_pokemon_state": b,
            "p1_move_details": None, "p2_move_details": None}


def battle(team, lead, timeline):
    return {"p1_team_details": [{"name": n} for n in team],
            "p2_lead_details": {"name": lead}, "battle_timeline": timeline}


class FeatureRegressions(unittest.TestCase):
    def test_empty_damage_has_zero_hits(self):
        b = battle(["a"], "b", [row(state("a"), state("b"))] * 2)
        features = move_effect_features(b)
        for side in ("p1", "p2"):
            for name in ("n_hits", "dmg_mean", "dmg_total", "dmg_max"):
                self.assertEqual(features[f"{side}_{name}"], 0)

    def test_unobserved_lead_constrains_team(self):
        catalog = [battle(["a", "b"], "a", []),
                   battle(["a", "c"], "a", [])]
        context = {"teams": build_team_index(catalog)}
        query = battle(["a", "b"], "b", [row(state("a"), state("a"))])
        match = team_id_features(query, context)
        self.assertEqual(match["p2_team_candidates"], 1)
        self.assertEqual(match["p2_team_id"], context["teams"]["ids"][("a", "b")])

    def test_fainted_opponent_is_not_reintroduced(self):
        b = battle(["a"], "b", [row(state("a"), state("b", 0.0, "fnt")),
                                  row(state("a"), state("c"))])
        context = {"teams": build_team_index([battle(["b", "c"], "b", [])]),
                   "matchups": {"edge": {("a", "b"): -20.0, ("a", "c"): 20.0},
                                "turns": {("a", "b"): 10, ("a", "c"): 10}}}
        before = deepcopy(b)
        f = win_condition_features(b, context)
        self.assertEqual(f["p1_unanswered"], 1)
        self.assertEqual(f["p1_wincon_best"], 1.0)
        self.assertEqual(b, before)


if __name__ == "__main__":
    unittest.main()
