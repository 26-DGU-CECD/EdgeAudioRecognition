"""End-to-end replay evaluator test with known artificial *scores*, not sound accuracy."""
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np


class ReplayEvaluatorTest(unittest.TestCase):
    def test_input_hops_and_alignment(self):
        with tempfile.TemporaryDirectory(prefix="audio-policy-pilot-",
                                         dir=Path(__file__).resolve().parent) as directory:
            assert Path(directory).resolve().is_relative_to(Path(__file__).resolve().parent)
            out = Path(directory)
            config = {
                "class_names": ["background", "gun", "siren"],
                "weights": (np.ones((3, 3)) / 3).tolist(),
                "global_threshold": .6,
                "class_thresholds": {"gun": .6, "siren": .6},
            }
            manifest = {
                "sample_rate": 16000, "window_seconds": 2.,
                "base_hop_seconds": .5, "warmup_seconds": 2.,
                "tail_seconds": 2.,
                "annotation": "Test fixture only",
                "episodes": [
                    {"episode_id": 0, "label": "gun", "label_id": 1, "group_id": "a"},
                    {"episode_id": 1, "label": "background", "label_id": 0, "group_id": "b"},
                ],
            }
            (out/"policy_config.json").write_text(json.dumps(config), encoding="utf-8")
            (out/"replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            fingerprint = hashlib.sha256(
                (out/"policy_config.json").read_bytes()
                + (out/"replay_manifest.json").read_bytes()).hexdigest()
            keys, scores = [], []
            for eid in (0, 1):
                for end in range(32000, 96001, 8000):
                    keys.append(f"{eid}:{end}")
                    foreground = eid == 0 and 32000 < end < 96000
                    scores.append([0., 8., 0.] if foreground else [8., 0., 0.])
            for m in ("efficientat", "yamnet", "mobilenetv4"):
                order = np.arange(len(keys))
                if m == "yamnet":
                    order = order[::-1]
                np.savez_compressed(
                    out/f"replay_{m}.npz", keys=np.asarray(keys)[order],
                    logits=np.asarray(scores)[order],
                    preprocessing_s=np.full(len(keys), .001),
                    inference_s=np.full(len(keys), .002),
                    fingerprint=np.asarray(fingerprint),
                )
            script = Path(__file__).parent / "evaluate_replay.py"
            result = subprocess.run([sys.executable, str(script), "--out", str(out)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            import csv
            with (out/"replay_policy_comparison.csv").open(
                    encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            expected = {"I0": (2., 6), "I1": (1., 10), "I2": (.5, 18)}
            for row in rows:
                first, windows = expected[row["input_policy"]]
                self.assertEqual(float(row["foreground_hit_by_insert_2s_audio_time"]), 1.)
                self.assertEqual(float(row["first_hit_audio_time_s_p50_detected_only"]), first)
                self.assertEqual(float(row["background_episode_any_alert_rate"]), 0.)
                self.assertEqual(int(row["scheduled_windows"]), windows)


if __name__ == "__main__":
    unittest.main()

