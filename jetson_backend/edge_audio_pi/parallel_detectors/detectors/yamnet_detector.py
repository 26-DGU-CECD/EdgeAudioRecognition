"""
YAMNet detector (TensorFlow Hub). Two modes:

  * zero-shot : MAX sigmoid score over the AudioSet classes mapping to each target
                (siren/scream/crying), aggregated over frames.
  * trained   : freeze YAMNet, mean-pool its 1024-d frame embeddings, and run a
                small MLP head fine-tuned on the target group (see train_yamnet.py).
                This is the canonical YAMNet transfer-learning setup.

`run_inference.py` picks trained mode automatically when a checkpoint exists.
YAMNet input: 16 kHz mono float32 (exactly the dataset format — no resampling).
Runs in the `paradet` env (tensorflow + torch coexist).
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .. import config
from .base import BaseDetector, LogitsUnavailable


def _ensure_ssl_certs() -> None:
    """Point urllib at certifi's CA bundle so TF-Hub downloads don't fail with
    SSL CERTIFICATE_VERIFY_FAILED on python.org macOS builds."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass


def load_yamnet():
    """Load the YAMNet SavedModel from TF-Hub (with the SSL cert fix)."""
    _ensure_ssl_certs()
    import tensorflow_hub as hub
    print(f"[YAMNet] loading from {config.YAMNET_HUB_HANDLE} ...")
    return hub.load(config.YAMNET_HUB_HANDLE)


def yamnet_class_names(model) -> list[str]:
    """AudioSet display names in YAMNet's output order (from its class map asset)."""
    path = model.class_map_path().numpy().decode("utf-8")
    names = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):        # index,mid,display_name
            names.append(r["display_name"])
    return names


# Frame pooling. A 2 s clip yields 4 YAMNet frames; mean alone smears short events
# (scream), so max/std can be concatenated. The choice is recorded in the checkpoint
# so inference always pools the same way training did.
POOLINGS = {"mean": 1, "meanmax": 2, "meanmaxstd": 3}


def pool_frames(frames: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """[N, n_frames, 1024] -> [N, 1024 * k] pooled over the frame axis."""
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {list(POOLINGS)}")
    parts = [frames.mean(axis=1)]
    if pooling in ("meanmax", "meanmaxstd"):
        parts.append(frames.max(axis=1))
    if pooling == "meanmaxstd":
        parts.append(frames.std(axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def pooled_embeddings(model, waveforms_16k: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """Return pooled [N, 1024 * k] YAMNet embeddings for a batch of clips."""
    out = np.zeros((len(waveforms_16k), 1024 * POOLINGS[pooling]), dtype=np.float32)
    for i, wav in enumerate(waveforms_16k):
        _scores, embeddings, _spec = model(wav.astype(np.float32))
        out[i] = pool_frames(embeddings.numpy()[None, ...], pooling)[0]
    return out


class YamnetHead:
    """MLP head on top of pooled YAMNet embeddings. Built lazily to avoid importing
    torch when only zero-shot mode is used.

    `small` is the original 1-hidden-layer head; `mlp2` adds input normalisation,
    batch-norm and a second hidden layer, which the wider pooled inputs can support.
    The architecture name is stored in the checkpoint (`head_arch`); checkpoints
    written before that field existed are `small`.
    """

    @staticmethod
    def build(input_dim: int, num_classes: int, arch: str = "small"):
        import torch.nn as nn
        if arch == "small":
            return nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        if arch == "mlp2":
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        raise ValueError(f"unknown head arch {arch!r}")


class YAMNetDetector(BaseDetector):
    target_classes = list(config.DETECTOR_CLASSES["yamnet"])

    def __init__(self, frame_agg: str | None = None, ckpt=None, calibration=None):
        self.frame_agg = frame_agg or config.YAMNET_FRAME_AGG
        assert self.frame_agg in ("max", "mean")
        self.model = load_yamnet()

        if ckpt is not None:
            # ---- trained mode ----
            import torch
            self.mode = "trained"
            state = torch.load(ckpt, map_location="cpu")
            self.target_classes = state["target_classes"]
            # Pre-2026-07 checkpoints predate these fields and were mean-pool + small.
            self.pooling = state.get("pooling", "mean")
            self.head = YamnetHead.build(state["input_dim"], len(self.target_classes),
                                         arch=state.get("head_arch", "small"))
            self.head.load_state_dict(state["head_state"])
            self.head.eval()
            self._torch = torch
            print(f"[YAMNet] trained head loaded: {ckpt} | {self.target_classes} "
                  f"| pooling={self.pooling} arch={state.get('head_arch', 'small')}")
        else:
            # ---- zero-shot mode ----
            self.mode = "zeroshot"
            names = yamnet_class_names(self.model)
            name_to_idx = {n: i for i, n in enumerate(names)}
            self.class_indices = {}
            for cls in self.target_classes:
                idxs = [name_to_idx[dn] for dn in config.YAMNET_AUDIOSET_NAMES[cls]
                        if dn in name_to_idx]
                if not idxs:
                    raise ValueError(f"No AudioSet indices resolved for class {cls!r}")
                self.class_indices[cls] = idxs
            print(f"[YAMNet] zero-shot class->indices: {self.class_indices}")

        # Zero-shot has no recoverable logits: TF-Hub YAMNet emits post-sigmoid scores
        # and 'mean' frame aggregation is not monotone-invertible. So nothing to
        # calibrate there either.
        self.provides_logits = self.mode == "trained"
        self._init_calibration("yamnet", calibration if self.provides_logits else False,
                               ckpt)

    def predict_logits_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        if self.mode != "trained":
            raise LogitsUnavailable(
                "YAMNet zero-shot returns TF-Hub post-sigmoid scores; there are no "
                "logits to recover.")
        emb = pooled_embeddings(self.model, waveforms_16k, self.pooling)   # [N, 1024*k]
        with self._torch.no_grad():
            return self.head(self._torch.from_numpy(emb)).numpy().astype(np.float32)

    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        if self.mode == "trained":
            return self.probs_from_logits(self.predict_logits_batch(waveforms_16k))

        # zero-shot
        out = np.zeros((len(waveforms_16k), len(self.target_classes)), dtype=np.float32)
        agg = np.max if self.frame_agg == "max" else np.mean
        for i, wav in enumerate(waveforms_16k):
            scores, _embeddings, _spec = self.model(wav.astype(np.float32))
            clip_scores = agg(scores.numpy(), axis=0)                   # [521]
            for j, cls in enumerate(self.target_classes):
                # batch-safe: inside the per-clip loop; reduces over this class's
                # AudioSet indices for one clip, never across clips.
                out[i, j] = clip_scores[self.class_indices[cls]].max()
        return out
