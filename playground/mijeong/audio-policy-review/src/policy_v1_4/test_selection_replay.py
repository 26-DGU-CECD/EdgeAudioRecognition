"""Known-score integration tests; no acoustic accuracy or hardware timing claim."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class SelectionReplayTests(unittest.TestCase):
    def make_fixture(self, out):
        names = ["background", "gun", "siren", "water"]
        owners = {"efficientat": ["gun"], "yamnet": ["siren"], "mobilenetv4": ["water"]}
        config = dict(class_names=names, scope="test fixture only",
                      selection_spec=dict(experiment_id="fixture", expert_classes=owners,
                                          model_class_names={m: names for m in owners}, score_type="softmax_logits"),
                      policies={"global_max": dict(global_threshold=.6,
                                                   class_thresholds={"gun": .6, "siren": .6, "water": .6})})
        episodes = [dict(episode_id=0, label="gun", label_id=1, group_id="same_recording", clip_insert_seconds=2., phase_offset_seconds=0.),
                    dict(episode_id=1, label="gun", label_id=1, group_id="same_recording", clip_insert_seconds=2.5, phase_offset_seconds=.5),
                    dict(episode_id=2, label="background", label_id=0, group_id="negative", clip_insert_seconds=2., phase_offset_seconds=0.)]
        manifest = dict(sample_rate=100, window_seconds=2., base_hop_seconds=.5,
                        warmup_seconds=2., tail_seconds=2., config_file="selection_config.json",
                        annotation="Known artificial score fixture", episodes=episodes)
        (out / "selection_config.json").write_text(json.dumps(config), encoding="utf-8")
        (out / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        fingerprint = hashlib.sha256((out / "selection_config.json").read_bytes()
                                     + (out / "replay_manifest.json").read_bytes()).hexdigest()
        for model in owners:
            keys, values = [], []
            for e in episodes:
                for end in range(200, round((e["clip_insert_seconds"] + 4.) * 100) + 1, 50):
                    keys.append(f"{e['episode_id']}:{end}")
                    intersects = end > e["clip_insert_seconds"] * 100 and end - 200 < (e["clip_insert_seconds"] + 2.) * 100
                    fg = e["label"] == "gun" and intersects
                    if fg:
                        # A stronger NONOWNER gun output in other models must
                        # not alter the owner's score or be fused with it.
                        values.append([0., 9. if model == "efficientat" else 12., 0., 0.])
                    else:
                        values.append([9., 0., 0., 0.])
            order = np.arange(len(keys))
            if model == "yamnet":
                order = order[::-1]
            np.savez_compressed(out / f"replay_{model}.npz", keys=np.asarray(keys)[order],
                                logits=np.asarray(values)[order],
                                preprocessing_s=np.full(len(keys), .001), inference_s=np.full(len(keys), .002),
                                fingerprint=np.asarray(fingerprint), head_class_names=np.asarray(names),
                                checkpoint_sha256=np.asarray("fixture-only"))
        return owners

    def run_evaluator(self, out):
        return subprocess.run([sys.executable, str(Path(__file__).parent / "evaluate_selection_replay.py"),
                               "--out", str(out), "--background-mode", "global_max",
                               "--output-policies", "S0", "S1", "S2"],
                              capture_output=True, text=True, encoding="utf-8")

    def test_input_phase_alignment_and_no_voting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            self.make_fixture(out)
            r = self.run_evaluator(out)
            self.assertEqual(r.returncode, 0, r.stderr)
            with (out / "input_selection_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            expected = {"I0": (1.75, 9, 1.), "I1": (.75, 15, 2.), "I2": (.5, 28, 4.)}
            for row in rows:
                delay, windows, relative = expected[row["input_policy"]]
                self.assertAlmostEqual(float(row["first_hit_audio_time_s_p50_detected_only"]), delay)
                self.assertEqual(int(row["scheduled_windows"]), windows)
                self.assertEqual(float(row["steady_calls_relative_to_I0"]), relative)
                self.assertEqual(float(row["foreground_hit_by_insert_2s_audio_time"]), 1.)
                self.assertEqual(float(row["background_episode_any_alert_rate"]), 0.)
                self.assertEqual(int(row["foreground_recorded_groups"]), 1)
            with (out / "input_selection_per_episode.csv").open(encoding="utf-8-sig", newline="") as f:
                details = list(csv.DictReader(f))
            i0 = [r for r in details if r["input_policy"] == "I0" and r["output_policy"] == "S0" and r["label"] == "gun"]
            self.assertEqual([float(r["first_correct_audio_time_from_insert_s"]) for r in i0], [2., 1.5])

    def test_stale_config_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            self.make_fixture(out)
            path = out / "selection_config.json"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            result = self.run_evaluator(out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale export", result.stderr)

    def test_prepare_preserves_eval_groups_and_expands_phases(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            (out / "selection_config.json").write_text(json.dumps({"class_names": ["background", "gun"]}), encoding="utf-8")
            paths = []
            for i in range(3):
                p = out / f"placeholder_{i}.wav"
                p.write_bytes(b"not-read-in-manifest-preparation")
                paths.append(str(p))
            np.savez_compressed(out / "selection_samples.npz", paths=np.asarray(paths), y_true=np.array([0, 1, 1]),
                                group_ids=np.array(["bg_eval", "gun_eval", "calibration_only"]),
                                evaluation_mask=np.array([True, True, False]))
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "prepare_replay.py"),
                                "--selection", "--out", str(out), "--max-per-class", "1"],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr)
            manifest = json.loads((out / "replay_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["episodes"]), 8)
            self.assertEqual(manifest["recorded_groups"], 2)
            self.assertEqual({e["group_id"] for e in manifest["episodes"]}, {"bg_eval", "gun_eval"})
            self.assertEqual({e["clip_insert_seconds"] for e in manifest["episodes"]}, {2., 2.5, 3., 3.5})


if __name__ == "__main__":
    unittest.main()
