from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from common_audio import AudioConfig, load_and_preprocess
from common_metadata import load_metadata, read_class_names
from evaluation_utils import save_classification_artifacts
import config


class Background20Sequence(tf.keras.utils.Sequence):
    """Exact 20% background with source-group-balanced sampling."""

    def __init__(
        self,
        x,
        y,
        group_ids,
        background_id: int,
        batch_size: int = 40,
        seed: int = 42,
    ):
        super().__init__()
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.group_ids = np.asarray(group_ids).astype(str)
        self.batch_size = int(batch_size)
        self.n_bg = int(round(self.batch_size * 0.20))
        self.n_fg = self.batch_size - self.n_bg
        self.seed = int(seed)
        self.epoch = 0

        self.bg_groups = {}
        self.fg_groups = {}
        for i, (label, group) in enumerate(zip(self.y, self.group_ids)):
            target = (
                self.bg_groups
                if int(label) == int(background_id)
                else self.fg_groups
            )
            target.setdefault(str(group), []).append(i)
        if not self.bg_groups or not self.fg_groups:
            raise ValueError("Both background and foreground groups are required")
        self._reset()

    @staticmethod
    def _cycle(groups, n, rng):
        out = []
        groups = list(groups)
        while len(out) < n:
            block = groups.copy()
            if len(block) > 1:
                rng.shuffle(block)
            out.extend(block)
        return out[:n]

    def _reset(self):
        self.rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        n_batches = len(self)
        self.fg_epoch_groups = self._cycle(
            self.fg_groups,
            n_batches * self.n_fg,
            self.rng,
        )
        self.bg_epoch_groups = self._cycle(
            self.bg_groups,
            n_batches * self.n_bg,
            self.rng,
        )

    def __len__(self):
        return math.ceil(len(self.fg_groups) / self.n_fg)

    def __getitem__(self, idx):
        fg_names = self.fg_epoch_groups[
            idx * self.n_fg : (idx + 1) * self.n_fg
        ]
        bg_names = self.bg_epoch_groups[
            idx * self.n_bg : (idx + 1) * self.n_bg
        ]
        fg_idx = [self.rng.choice(self.fg_groups[g]) for g in fg_names]
        bg_idx = [self.rng.choice(self.bg_groups[g]) for g in bg_names]
        indices = np.asarray(fg_idx + bg_idx, dtype=np.int64)
        self.rng.shuffle(indices)
        return self.x[indices], self.y[indices]

    def on_epoch_end(self):
        self._reset()


def _load_shared_embeddings(
    shared_path: Path,
    requested_df: pd.DataFrame,
    label_to_id: dict[str, int],
):
    if not shared_path.is_file():
        return None
    data = np.load(shared_path, allow_pickle=False)
    if not {"x", "paths"}.issubset(data.files):
        return None
    shared_paths = np.asarray(data["paths"]).astype(str)
    index = {path: i for i, path in enumerate(shared_paths)}
    requested_paths = requested_df["path"].astype(str).tolist()
    if any(path not in index for path in requested_paths):
        return None
    order = np.asarray([index[path] for path in requested_paths], dtype=np.int64)
    x = np.asarray(data["x"])[order]
    y = np.asarray(
        [label_to_id[str(v)] for v in requested_df["label"]],
        dtype=np.int64,
    )
    paths = np.asarray(
        requested_paths,
        dtype=np.str_,
    )

    groups = np.asarray(
        requested_df["group_id"].astype(str).tolist(),
        dtype=np.str_,
    )
    return x, y, paths, groups


def extract_split(
    csv_path,
    split,
    class_names,
    audio_cfg,
    yamnet,
    cache_path: Path,
    reuse_cache_dir: Path | None = None,
):
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if {"x", "y", "paths", "group_ids"}.issubset(data.files):
            return data["x"], data["y"], data["paths"], data["group_ids"]

    df = load_metadata(csv_path, split)
    label_to_id = {name: i for i, name in enumerate(class_names)}

    if reuse_cache_dir is not None:
        shared = _load_shared_embeddings(
            reuse_cache_dir / f"{split}.npz",
            df,
            label_to_id,
        )
        if shared is not None:
            x, y, paths, groups = shared
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                x=x,
                y=y,
                paths=paths,
                group_ids=groups,
            )
            print(f"{split}: reused embeddings from {reuse_cache_dir}")
            return x, y, paths, groups

    if yamnet is None:
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    x_list, y_list, paths, group_ids = [], [], [], []
    for i, row in df.iterrows():
        wav, _, _ = load_and_preprocess(row["path"], audio_cfg)
        _, embeddings, _ = yamnet(
            tf.convert_to_tensor(wav, dtype=tf.float32)
        )
        clip_embedding = (
            tf.reduce_mean(embeddings, axis=0)
            .numpy()
            .astype(np.float32)
        )
        x_list.append(clip_embedding)
        y_list.append(label_to_id[str(row["label"])])
        paths.append(str(row["path"]))
        group_ids.append(str(row["group_id"]))
        if (i + 1) % 100 == 0:
            print(f"{split}: {i + 1}/{len(df)} embeddings")

    x = np.stack(x_list)
    y = np.asarray(y_list, dtype=np.int64)
    paths_arr = np.asarray(
        paths,
        dtype=np.str_,
    )

    groups_arr = np.asarray(
        group_ids,
        dtype=np.str_,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        paths=paths_arr,
        group_ids=groups_arr,
    )
    return x, y, paths_arr, groups_arr


def build_head(num_classes: int):
    inputs = tf.keras.Input(shape=(1024,), dtype=tf.float32)
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.30)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None)(x)
    return tf.keras.Model(inputs, logits, name="yamnet_embedding_head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(config.METADATA_CSV))
    parser.add_argument("--classes", default=str(config.CLASS_NAMES_PATH))
    parser.add_argument("--output", default=str(config.YAMNET_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=config.HIGHPASS_HZ,
    )
    parser.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=config.SILENCE_THRESHOLD_DBFS,
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=config.TARGET_RMS_DBFS,
    )
    parser.add_argument(
        "--reuse-cache-dir",
        default=None,
        help="Existing YAMNet embedding_cache directory.",
    )
    args = parser.parse_args()
    config.validate_paths()

    tf.keras.utils.set_random_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(args.classes)
    bg_id = class_names.index("background")
    audio_cfg = AudioConfig(
        highpass_hz=args.highpass_hz,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        target_rms_dbfs=args.target_rms_dbfs,
    )

    cache_dir = output / "embedding_cache"
    reuse_cache_dir = (
        Path(args.reuse_cache_dir)
        if args.reuse_cache_dir
        else None
    )

    yamnet = None
    x_train, y_train, train_paths, train_groups = extract_split(
        args.csv,
        "train",
        class_names,
        audio_cfg,
        yamnet,
        cache_dir / "train.npz",
        reuse_cache_dir,
    )
    x_val, y_val, val_paths, val_groups = extract_split(
        args.csv,
        "val",
        class_names,
        audio_cfg,
        yamnet,
        cache_dir / "val.npz",
        reuse_cache_dir,
    )
    x_test, y_test, test_paths, test_groups = extract_split(
        args.csv,
        "test",
        class_names,
        audio_cfg,
        yamnet,
        cache_dir / "test.npz",
        reuse_cache_dir,
    )

    train_seq = Background20Sequence(
        x_train,
        y_train,
        train_groups,
        bg_id,
        args.batch_size,
        args.seed,
    )
    model = build_head(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output / "best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(output / "history.csv"),
    ]
    model.fit(
        train_seq,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    val_logits = model.predict(
        x_val,
        batch_size=args.batch_size,
        verbose=0,
    )
    test_logits = model.predict(
        x_test,
        batch_size=args.batch_size,
        verbose=0,
    )
    save_classification_artifacts(
        output,
        "val",
        y_val,
        val_logits,
        val_paths,
        class_names,
    )
    save_classification_artifacts(
        output,
        "test",
        y_test,
        test_logits,
        test_paths,
        class_names,
    )


if __name__ == "__main__":
    main()
