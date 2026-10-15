import json
import tempfile
from pathlib import Path
import unittest
from project_io import (
    config_experiment_paths, read_transfer_policy, validate_experiment_spec,
)


class ExperimentExtensionTests(unittest.TestCase):
    names = ["background", "gun", "siren", "water"]

    def spec(self):
        return {
            "experiment_id": "example_selected", "assignment_type": "custom",
            "nonexpert_fraction": .7,
            "specialist_dir": "runs/example_specialists",
            "ensemble_dir": "runs/example_ensemble",
            "expert_classes": {
                "efficientat": ["gun"], "yamnet": ["siren"], "mobilenetv4": ["water"],
            },
        }

    def test_custom_paths_override_legacy_layout(self):
        spec = self.spec()
        config = {
            "class_names": self.names, "experiment_spec": spec,
            "assignment": "custom", "ratio": .7,
        }
        self.assertEqual(config_experiment_paths(config),
                         ("runs/example_specialists", "runs/example_ensemble"))

    def test_legacy_config_remains_supported(self):
        self.assertEqual(
            config_experiment_paths({"assignment": "predefined", "ratio": .7}),
            ("runs/E_predefined_specialists/nonexpert_070",
             "runs/F_predefined_weighted_ensemble/nonexpert_070"),
        )

    def test_overlapping_ownership_rejected(self):
        spec = self.spec()
        spec["expert_classes"]["yamnet"].append("gun")
        with self.assertRaises(ValueError):
            validate_experiment_spec(spec, self.names)

    def test_template_and_path_escape_rejected(self):
        spec = self.spec()
        spec["ready_for_evaluation"] = False
        with self.assertRaises(ValueError):
            validate_experiment_spec(spec, self.names)
        spec = self.spec()
        spec["specialist_dir"] = "../outside"
        with self.assertRaises(ValueError):
            validate_experiment_spec(spec, self.names)

    def test_transfer_keeps_parameters_but_never_old_weights(self):
        with tempfile.TemporaryDirectory(
                prefix="policy-transfer-test-", dir=Path(__file__).resolve().parent) as d:
            path = Path(d).resolve()
            assert path.is_relative_to(Path(__file__).resolve().parent)
            config = {
                "class_names": self.names, "seed": 42,
                "weights": [[999]], "experiment_spec": {"old": "model"},
                "global_threshold": .6,
                "class_thresholds": {"gun": .5, "siren": None, "water": .7},
                "target_precision": .95, "minimum_accepted": 20,
            }
            file = path/"policy.json"
            file.write_text(json.dumps(config), encoding="utf-8")
            policy = read_transfer_policy(file, self.names, 42)
            self.assertEqual(policy["class_thresholds"], config["class_thresholds"])
            self.assertNotIn("weights", policy)
            self.assertNotIn("experiment_spec", policy)
            with self.assertRaises(ValueError):
                read_transfer_policy(file, list(reversed(self.names)), 42)
            with self.assertRaises(ValueError):
                read_transfer_policy(file, self.names, 43)


if __name__ == "__main__":
    unittest.main()

