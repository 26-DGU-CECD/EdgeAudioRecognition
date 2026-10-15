import unittest
import numpy as np
from selection_core import owned_candidates, decide, subset, fit_thresholds
from policy_core import softmax


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.names = ["background", "a", "b", "c"]
        self.experts = {"x": ["a"], "y": ["b"], "z": ["c"]}
        self.heads = {m: self.names for m in self.experts}
        # Each head sums to 1; nonowner outputs must never win or be averaged.
        self.outputs = {"x": np.array([[.1, .2, .65, .05]]),
                        "y": np.array([[.1, .4, .3, .2]]),
                        "z": np.array([[.1, .1, .1, .7]])}

    def candidates(self, mode="global_max", outputs=None):
        return owned_candidates(outputs or self.outputs, self.names, self.experts,
                                self.heads, mode, "probabilities")

    def test_owner_only_max_without_renormalization(self):
        c = self.candidates()
        np.testing.assert_allclose(c.scores, [[.1, .2, .3, .7]])
        self.assertNotAlmostEqual(c.scores.sum(), 1.)
        d = decide(c)
        self.assertEqual(int(d["top_id"][0]), 3)
        self.assertAlmostEqual(float(d["score"][0]), .7)

    def test_heterogeneous_heads_and_full_head_softmax(self):
        heads = {"x": ["background", "a", "b"], "y": ["b", "background"],
                 "z": ["c", "background"]}
        z = {"x": np.array([[0., 1., 5.]]), "y": np.array([[1., 0.]]),
             "z": np.array([[2., 0.]])}
        c = owned_candidates(z, self.names, self.experts, heads, "exclude")
        self.assertAlmostEqual(c.scores[0, 1], softmax(z["x"])[0, 1])
        self.assertLess(c.scores[0, 1], .03)  # Not softmax([bg, owned]) ~.73.
        self.assertEqual(int(decide(c)["top_id"][0]), 3)

    def test_background_variants_are_distinct(self):
        outputs = {"x": np.array([[.8, .1, .05, .05]]),
                   "y": np.array([[.4, .1, .45, .05]]),
                   "z": np.array([[.8, .05, .05, .1]])}
        self.assertEqual(int(decide(self.candidates(outputs=outputs))["top_id"][0]), 0)
        self.assertEqual(int(decide(self.candidates("exclude", outputs))["top_id"][0]), 2)
        # Other models' strong backgrounds cannot veto y after local gating.
        self.assertEqual(int(decide(self.candidates("per_model_gate", outputs))["top_id"][0]), 2)

    def test_all_gated_returns_background(self):
        outputs = {m: np.array([[.85, .05, .05, .05]]) for m in self.experts}
        d = decide(self.candidates("per_model_gate", outputs))
        self.assertEqual(d["status"][0], "background")
        self.assertFalse(d["accepted"][0])

    def test_no_fallback_after_top_rejection(self):
        d = decide(self.candidates(), "S2", class_thresholds={"a": 0., "b": 0., "c": .8})
        self.assertEqual(d["top_id"][0], 3)
        self.assertFalse(d["accepted"][0])
        self.assertEqual(d["status"][0], "unconfirmed")

    def test_disabled_common_and_class_thresholds(self):
        self.assertFalse(decide(self.candidates(), "S1", None)["accepted"][0])
        d = decide(self.candidates(), "S2", class_thresholds={"a": None, "b": None, "c": None})
        self.assertFalse(d["accepted"][0])

    def test_bad_ownership_and_missing_background_fail(self):
        experts = {"x": ["a", "b"], "y": ["b"], "z": ["c"]}
        with self.assertRaises(ValueError):
            owned_candidates(self.outputs, self.names, experts, self.heads, "global_max", "probabilities")
        heads = {"x": ["a"], "y": ["b"], "z": ["c"]}
        with self.assertRaises(ValueError):
            owned_candidates({m: np.array([[1.]]) for m in heads}, self.names, self.experts,
                             heads, "global_max")

    def test_fit_only_uses_supplied_calibration_subset(self):
        outputs = {m: np.repeat(p, 3, axis=0) for m, p in self.outputs.items()}
        c = self.candidates(outputs=outputs)
        y = np.array([3, 3, 1])
        common, by_class = fit_thresholds(subset(c, [True, True, False]), y[:2], 1., 2)
        self.assertEqual(common, 0.)
        self.assertEqual(by_class["c"], 0.)
        self.assertIsNone(by_class["a"])
        common_all, _ = fit_thresholds(c, y, 1., 2)
        self.assertIsNone(common_all)

    def test_ties_use_class_order(self):
        outputs = {m: np.full((1, 4), .25) for m in self.experts}
        self.assertEqual(decide(self.candidates(outputs=outputs))["status"][0], "background")
        self.assertEqual(int(decide(self.candidates("exclude", outputs))["top_id"][0]), 1)


if __name__ == "__main__":
    unittest.main()
