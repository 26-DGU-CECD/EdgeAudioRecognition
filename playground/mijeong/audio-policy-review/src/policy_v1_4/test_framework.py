import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import hashlib
import numpy as np
from framework_core import (common_probabilities, fit_class_weights, fusion_candidates,
                            fit_acceptance, decide_rule, certify_partial, AdaptiveHop, TemporalConfirmation,
                            PartialVoteCoordinator, AdaptiveAudioBuffer)
from selection_core import CandidateScores
from run_framework_pilot import group_split
from evaluate_framework_replay import evaluate_episode
import test_selection_replay as selection_fixture


class FrameworkCoreTests(unittest.TestCase):
    def setUp(self):
        self.names = ["background", "gun", "siren", "water"]
        self.experts = {"x": ["gun"], "y": ["siren"], "z": ["water"]}
        self.heads = {m: self.names for m in self.experts}

    def test_voting_and_owner_max_are_different(self):
        outputs = {"x": np.array([[.05, .6, .3, .05]]),
                   "y": np.array([[.05, .3, .6, .05]]),
                   "z": np.array([[.05, .85, .05, .05]])}
        owner = fusion_candidates(outputs, self.names, self.experts, self.heads, "F0", "probabilities")
        voted = fusion_candidates(outputs, self.names, self.experts, self.heads, "F1", "probabilities")
        np.testing.assert_allclose(owner.scores, [[.05, .6, .6, .05]])
        np.testing.assert_allclose(voted.scores, np.stack(list(outputs.values())).mean(axis=0))
        self.assertAlmostEqual(voted.scores.sum(), 1.)
        self.assertNotAlmostEqual(owner.scores.sum(), 1.)

    def test_missing_head_classes_not_zero_padded(self):
        with self.assertRaisesRegex(ValueError, "SAME complete"):
            common_probabilities({"x": np.array([[0., 1.]])}, self.names,
                                 {"x": ["background", "gun"]}, "softmax_logits")

    def test_f2_weights_positive_and_column_normalized(self):
        p = np.array([[[.1, .9, 0., 0.], [.1, 0., .9, 0.]],
                      [[.1, 0., .9, 0.], [.1, 0., .9, 0.]],
                      [[.1, 0., 0., .9], [.1, 0., 0., .9]]])
        weights = fit_class_weights(p, np.array([1, 2]), self.names, list(self.experts), self.experts)
        self.assertTrue((weights > 0).all())
        np.testing.assert_allclose(weights.sum(axis=0), 1.)
        self.assertGreater(weights[0, 1], weights[1, 1])

    def test_r2_precision_failure_is_not_null_label_ban(self):
        names = ["background", "fire_alarm", "siren"]
        scores = np.array([[.1, .7, .2]] * 40 + [[.05, .05, .9]] * 40)
        truth = np.array([1] * 20 + [0] * 20 + [2] * 40)
        c = CandidateScores(scores, np.ones_like(scores, dtype=bool), tuple(names))
        fit = fit_acceptance(c, truth, .95, 20, .5)
        self.assertGreater(fit["global_threshold"], .7)
        self.assertLessEqual(fit["class_thresholds"]["fire_alarm"], .7)
        alarm = next(r for r in fit["diagnostics"] if r["label"] == "fire_alarm")
        self.assertEqual(alarm["threshold_source"], "recall_backoff_precision_relaxed")
        self.assertFalse(alarm["target_precision_met"])
        self.assertTrue(alarm["recall_floor_met"])

    def test_three_group_splits_preserve_previous_eval_half(self):
        groups = np.array([f"recording_{i}" for i in range(100)] * 2)
        masks = group_split(groups, 42)
        prior_eval = np.array([int(hashlib.sha256(f"42|{g}".encode()).hexdigest()[:16], 16) % 2 != 0 for g in groups])
        np.testing.assert_array_equal(masks["evaluation"], prior_eval)
        sets = [set(groups[m]) for m in masks.values()]
        self.assertFalse(any(a & b for i, a in enumerate(sets) for b in sets[i+1:]))

    def test_label_recall_floors_and_zero_sample_hop_guard(self):
        p = np.array([[.05, .9, .03, .02]] * 20)
        fit = fit_acceptance(CandidateScores(p, np.ones_like(p, dtype=bool), tuple(self.names)),
                             np.ones(20, dtype=int), recall_floors={"gun": .9})
        self.assertEqual(next(r for r in fit["diagnostics"] if r["label"] == "gun")["recall_floor"], .9)
        with self.assertRaisesRegex(ValueError, "adaptive hops"):
            AdaptiveAudioBuffer(self.names, sample_rate=100, policy=AdaptiveHop(fast=.00001))

    def test_adaptive_and_temporal_policies(self):
        hop = AdaptiveHop()
        self.assertEqual(hop.choose([.01, .01, .01, .97], self.names), 1.)
        self.assertEqual(hop.choose([.01, .01, .97, .01], self.names), .5)
        self.assertEqual(hop.choose([.1, .3, .3, .3], self.names), .5)
        delayed = TemporalConfirmation(self.names, "T1")
        self.assertIsNone(delayed.step(2., 2., [.1, .7, .1, .1], 1, True, lambda c: .6)[0])
        self.assertEqual(delayed.step(4., 4., [.1, .7, .1, .1], 1, True, lambda c: .6)[0], 1)
        event = TemporalConfirmation(self.names, "T2")
        self.assertEqual(event.step(2., 2., [.1, .7, .1, .1], 1, True, lambda c: .6)[0], 1)
        event = TemporalConfirmation(self.names, "T2")
        self.assertIsNone(event.step(2., 2., [.1, .1, .7, .1], 2, True, lambda c: .6)[0])
        self.assertEqual(event.step(2.5, 2.5, [.1, .1, .7, .1], 2, True, lambda c: .6)[0], 2)
        self.assertEqual(event.step(3., 3., [.15, .1, .6, .15], 2, False, lambda c: .7)[0], 2)
        self.assertIsNone(event.step(3.5, 3.5, [.8, .05, .1, .05], 0, False, lambda c: .7)[0])
        self.assertEqual(event.step(6., 3., [.01, .01, .97, .01], 2, True, lambda c: .7)[1], "stale_result")

    def test_partial_certificate_with_missing_data_nan(self):
        names = ["background", "gun", "siren"]
        p = np.array([[.01, .95, .04], [.01, .9, .09], [np.nan] * 3])
        weights = np.full((3, 3), 1/3)
        fitted = dict(global_threshold=.55, class_thresholds={"gun": .55, "siren": .55})
        cert = certify_partial(p, [True, True, False], weights, False, names, "R1", fitted)
        self.assertEqual(cert["top_id"], 1)
        self.assertTrue(cert["accepted"])
        fitted["global_threshold"] = .7
        self.assertIsNone(certify_partial(p, [True, True, False], weights, False, names, "R1", fitted))

    def test_random_certificates_match_full_vote_all_arrival_orders(self):
        rng = np.random.default_rng(22)
        names = ["background", "gun", "siren"]
        for _ in range(60):
            p = rng.dirichlet([.2, .2, .2], size=3)
            for normalized in (False, True):
                w = rng.uniform(.1, 1., size=(3, 3)) if normalized else np.ones((3, 3))
                w /= w.sum(axis=0)
                full = (p * w).sum(axis=0)
                if normalized:
                    full /= full.sum()
                c = CandidateScores(full[None], np.ones((1, 3), dtype=bool), tuple(names))
                fit = dict(global_threshold=.6, class_thresholds={"gun": .4, "siren": .7})
                for rule in ("R0", "R1", "R2"):
                    oracle = decide_rule(c, rule, fit)
                    for permutation in itertools.permutations(range(3)):
                        ready, visible = np.zeros(3, dtype=bool), np.full((3, 3), np.nan)
                        for i in permutation:
                            ready[i], visible[i] = True, p[i]
                            result = certify_partial(visible, ready, w, normalized, names, rule, fit)
                            if result is not None:
                                self.assertEqual(result["top_id"], int(oracle["top_id"][0]))
                                self.assertEqual(result["accepted"], bool(oracle["accepted"][0]))

    def test_asynchronous_completion_does_not_revert_to_old_window(self):
        names = ["background", "gun", "siren"]
        p = np.array([[[.95, .025, .025]] + [[.025, .95, .025]] * 3,
                      [[.025, .95, .025]] * 4, [[.025, .95, .025]] * 4])
        full = p.mean(axis=0)
        c = CandidateScores(full, np.ones_like(full, dtype=bool), tuple(names))
        fit = dict(global_threshold=.5, class_thresholds={"gun": .5, "siren": .5})
        d = decide_rule(c, "R0", fit)
        e = dict(episode_id=0, label="gun", label_id=1, group_id="r", clip_insert_seconds=2.)
        row, trace = evaluate_episode(e, np.arange(4), np.array([200, 250, 300, 350]), c, d,
            np.array([[.1]*4, [.2]*4, [4.]*4]), 100, 50, "I2", "independent_worker_sim", "T0", names,
            "R0", fit, AdaptiveHop(), {}, p, np.full((3, 3), 1/3), False, "bounded")
        self.assertEqual(row["superseded_output_windows"], 1)
        self.assertAlmostEqual(row["first_correct_from_clip_s"], .7)
        self.assertEqual(row["scheduled_model_calls"], 12)
        self.assertTrue(all(a["decision_time_s"] <= b["decision_time_s"] for a, b in zip(trace, trace[1:])))

    def test_callback_coordinator_seals_certified_window_and_drops_old(self):
        fit = dict(global_threshold=.5, class_thresholds={"gun": .5, "siren": .5, "water": .5})
        cfg = dict(class_names=self.names, selection_spec=dict(expert_classes=self.experts,
            model_class_names=self.heads, score_type="probabilities"), framework=dict(
            model_order=list(self.experts), fusions=["F1"], acceptance={"F1": fit}))
        coordinator = PartialVoteCoordinator(cfg, "F1", "R1")
        self.assertIsNone(coordinator.submit(1, "x", [.01, .95, .02, .02], 0, 32000))
        result = coordinator.submit(1, "y", [.01, .95, .02, .02], 0, 32000)
        self.assertEqual(result["display_label"], "gun")
        self.assertEqual(result["models_ready"], 2)
        self.assertFalse(result["model_jobs_cancelled"])
        self.assertIsNone(coordinator.submit(1, "z", [.95, .01, .02, .02], 0, 32000))
        self.assertEqual(coordinator.stale_results, 1)

    def test_callback_rejects_mixed_windows(self):
        fit = dict(global_threshold=.5, class_thresholds={"gun": .5, "siren": .5, "water": .5})
        cfg = dict(class_names=self.names, selection_spec=dict(expert_classes=self.experts,
            model_class_names=self.heads, score_type="probabilities"), framework=dict(
            model_order=list(self.experts), fusions=["F1"], acceptance={"F1": fit}))
        coordinator = PartialVoteCoordinator(cfg)
        coordinator.submit(1, "x", [.25]*4, 0, 32000)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            coordinator.submit(1, "y", [.25]*4, 8000, 40000)

    def test_audio_keeps_arriving_during_adaptive_inference(self):
        b = AdaptiveAudioBuffer(self.names, sample_rate=100)
        self.assertIsNone(b.push(np.arange(100)))
        window = b.push(np.arange(100, 200))
        self.assertEqual(window.end_sample, 200)
        self.assertIsNone(b.push(np.arange(200, 250)))
        # Strong urgent result chooses fast .5 and the new samples are already available.
        next_window = b.complete(window.window_id, [.01, .01, .97, .01])
        self.assertEqual(next_window.end_sample, 250)
        np.testing.assert_array_equal(next_window.waveform, np.arange(50, 250))
        self.assertLessEqual(len(b.buffer), 250)


class FrameworkReplayTests(unittest.TestCase):
    def test_new_evaluator_accepts_existing_export_format(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            selection_fixture.SelectionReplayTests().make_fixture(out)
            cfg_path = out / "selection_config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            fit = dict(global_threshold=.6, class_thresholds={"gun": .6, "siren": .6, "water": .6})
            cfg.update(schema="policy_framework_v1", framework=dict(
                model_order=list(cfg["selection_spec"]["expert_classes"]), fusions=["F0", "F1", "F2"],
                owner_background="global_max", classwise_weights=np.full((3, 4), 1/3).tolist(),
                acceptance={f: fit for f in ("F0", "F1", "F2")}))
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            fp = hashlib.sha256(cfg_path.read_bytes() + (out / "replay_manifest.json").read_bytes()).hexdigest()
            for model in cfg["framework"]["model_order"]:
                p = out / f"replay_{model}.npz"
                with np.load(p, allow_pickle=False) as z:
                    data = {k: z[k] for k in z.files}
                data["fingerprint"] = np.asarray(fp)
                np.savez_compressed(p, **data)
            result = subprocess.run([sys.executable, str(Path(__file__).parent / "evaluate_framework_replay.py"),
                "--out", str(out), "--inputs", "I0", "I1", "I2", "I3"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            import csv
            with (out / "framework_replay_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 12)
            baseline = next(r for r in rows if r["fusion"] == "F0" and r["input_policy"] == "I0")
            self.assertEqual(float(baseline["first_correct_s_p50_detected_only"]), 1.75)


if __name__ == "__main__":
    unittest.main()
