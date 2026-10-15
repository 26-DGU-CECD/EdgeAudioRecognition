import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from check_replay_audio import audit_audio
from export_replay_predictions import require_audio_preflight


class FakeAudioReader:
    def info(self, path):
        if path == "broken":
            raise RuntimeError("bad source")
        return SimpleNamespace(samplerate=16000, frames=16000, channels=2)

    def read(self, path, frames, dtype, always_2d):
        value = np.full((frames, 2), np.nan if path == "nonfinite" else 0., dtype=np.float32)
        return value, 16000


class PreflightTests(unittest.TestCase):
    def test_decode_errors_nonfinite_and_repeated_phases(self):
        episodes = [dict(path=p, label="gun", group_id="g")
                    for p in ("ok", "ok", "broken", "nonfinite")]
        rows = audit_audio({"episodes": episodes}, FakeAudioReader())
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["phase_episodes"], 2)
        self.assertTrue(rows[0]["shorter_than_2s"])
        self.assertEqual([r["status"] for r in rows[1:]], ["error", "error"])

    def test_matching_manifest_required_before_gpu_loading(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out, content = Path(tmp), b'{"episodes":[]}'
            with self.assertRaisesRegex(ValueError, "check_replay_audio"):
                require_audio_preflight(out, content)
            (out / "audio_preflight.json").write_text(json.dumps(
                dict(passed=True, manifest_sha256=hashlib.sha256(content).hexdigest())), encoding="utf-8")
            require_audio_preflight(out, content)
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                require_audio_preflight(out, content + b" ")


if __name__ == "__main__":
    unittest.main()
