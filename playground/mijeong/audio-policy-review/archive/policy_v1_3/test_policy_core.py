import itertools
import unittest
import numpy as np
from policy_core import (
    SlidingWindowScheduler, OutputPolicy, WindowFusionCoordinator,
    fit_class_thresholds, fuse_logits,
)


class PolicyTests(unittest.TestCase):
    def test_window_partition_and_values(self):
        signal = np.arange(64000, dtype=np.float32)
        windows = []
        scheduler = SlidingWindowScheduler(hop_seconds=.5)
        for chunk in np.array_split(signal, 137):
            windows.extend(scheduler.push(chunk))
        self.assertEqual([w.end_sample for w in windows], [32000, 40000, 48000, 56000, 64000])
        for window in windows:
            np.testing.assert_array_equal(
                window.waveform, signal[window.start_sample:window.end_sample])
            self.assertFalse(window.waveform.flags.writeable)
        whole = SlidingWindowScheduler(hop_seconds=.5).push(signal)
        for a, b in zip(windows, whole):
            np.testing.assert_array_equal(a.waveform, b.waveform)

    def test_three_input_counts(self):
        expected = {2.0: 30, 1.0: 59, .5: 117}
        for hop, count in expected.items():
            windows = SlidingWindowScheduler(hop_seconds=hop).push(
                np.zeros(60 * 16000, dtype=np.float32))
            self.assertEqual(len(windows), count)
            self.assertEqual(windows[-1].end_seconds, 60.0)

    def test_scheduler_rejects_bad_audio(self):
        scheduler = SlidingWindowScheduler()
        with self.assertRaises(ValueError):
            scheduler.push(np.zeros((20, 2)))
        with self.assertRaises(ValueError):
            scheduler.push(np.array([np.nan]))

    def test_fusion_class_normalization(self):
        z = np.array([[2., 0., 1.], [1., 3., 0.], [0., 1., 2.]])
        w = np.array([[.6, .2, .3], [.2, .5, .3], [.2, .3, .4]])
        prob = np.exp(z - z.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        raw = (prob * w).sum(axis=0)
        np.testing.assert_allclose(fuse_logits(z, w), raw / raw.sum())

    def test_global_top1_not_replaced(self):
        names = ["background", "gun", "siren"]
        p = np.array([[.1, .5, .4], [.7, .2, .1]])
        policy = OutputPolicy(names, "O2",
                              class_thresholds={"gun": .8, "siren": .3})
        result = policy.decide(p)
        self.assertEqual(result["top_id"].tolist(), [1, 0])
        self.assertEqual(result["status"].tolist(), ["unconfirmed", "background"])
        self.assertFalse(result["accepted"].any())

    def test_disabled_label_and_fit_requirements(self):
        names = ["background", "gun", "siren"]
        p = np.array([[.1, .8, .1], [.1, .2, .7], [.2, .7, .1]])
        y = np.array([1, 0, 1])
        thresholds, _ = fit_class_thresholds(
            p, y, names, target_precision=.95, minimum_accepted=2)
        self.assertIsNotNone(thresholds["gun"])
        self.assertIsNone(thresholds["siren"])
        result = OutputPolicy(names, "O2", class_thresholds=thresholds).decide(p)
        self.assertEqual(result["accepted"].tolist(), [True, False, True])

    def test_async_model_arrival_orders(self):
        names = ["background", "gun", "siren"]
        w = np.ones((3, 3)) / 3
        z = np.array([[0., 3., 1.], [1., 4., 0.], [0., 2., 1.]])
        model_names = ["efficientat", "yamnet", "mobilenetv4"]
        for order in itertools.permutations(range(3)):
            coordinator = WindowFusionCoordinator(w, names, OutputPolicy(names, "O0"))
            for position, i in enumerate(order):
                result = coordinator.submit(0, model_names[i], z[i], 0, 32000)
                if position < 2:
                    self.assertIsNone(result)
            np.testing.assert_allclose(result["probabilities"], fuse_logits(z, w))
            self.assertEqual(result["top_id"], 1)

    def test_async_windows_must_match(self):
        coordinator = WindowFusionCoordinator(
            np.ones((3, 3))/3, ["background", "gun", "siren"],
            OutputPolicy(["background", "gun", "siren"], "O0"))
        coordinator.submit(0, "efficientat", np.ones(3), 0, 32000)
        with self.assertRaises(ValueError):
            coordinator.submit(0, "yamnet", np.ones(3), 1, 32001)
        with self.assertRaises(ValueError):
            coordinator.submit(0, "efficientat", np.ones(3), 0, 32000)


if __name__ == "__main__":
    unittest.main()

