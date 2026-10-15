from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import config

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def read_classes(path: Path) -> list[str]:
    classes = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(classes) != len(set(classes)):
        raise ValueError("Duplicate class names")
    if "background" not in classes:
        raise ValueError("class_names.txt must contain background")
    return classes


def normalize_token(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def detect_label(relative_parts: tuple[str, ...], classes: list[str]) -> str | None:
    class_lookup = {normalize_token(c): c for c in classes}
    for part in relative_parts:
        key = normalize_token(part)
        if key in class_lookup:
            return class_lookup[key]
    return None


def detect_split(relative_parts: tuple[str, ...]) -> str | None:
    for part in relative_parts:
        key = normalize_token(part)
        if key in SPLIT_ALIASES:
            return SPLIT_ALIASES[key]
    return None


def infer_background_source(rel_path: Path, classes: list[str]) -> tuple[str, str]:
    """Best-effort origin class inference from background subfolders or filename.

    Returns (source_class, status). source_class is metadata only and never the
    training target. Longest class names are matched first so baby_cry wins over cry.
    """
    candidates = [c for c in classes if c != "background"]
    normalized_parts = [normalize_token(p) for p in rel_path.parts]
    normalized_text = "_".join(normalized_parts)

    # Exact path-component match is strongest.
    exact = []
    for c in candidates:
        token = normalize_token(c)
        if token in normalized_parts:
            exact.append(c)
    if len(exact) == 1:
        return exact[0], "exact_component"
    if len(exact) > 1:
        exact.sort(key=len, reverse=True)
        return exact[0], "ambiguous_component"

    # Then search token-bounded substrings in the whole relative path.
    matches = []
    for c in sorted(candidates, key=len, reverse=True):
        token = normalize_token(c)
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", normalized_text):
            matches.append(c)
    if len(matches) == 1:
        return matches[0], "token_match"
    if len(matches) > 1:
        return matches[0], "ambiguous_token"
    return "unknown", "unresolved"


def derive_group_id(path: Path, label: str, group_regex: str | None) -> str:
    """Derive a source-recording group from the filename.

    The group is intentionally independent of the training label. Therefore a
    low-volume segment moved to ``background`` and an audible segment kept in its
    original class can still share one group and one data split.

    Default convention: the stable base name is followed by an orig/original/aug
    marker and optional numbers, for example:
      source_name_orig.wav, source_name_orig_03.wav, source_name_03_aug_2.wav
    All of those become ``source_name`` or ``source_name_03`` depending on where
    the marker starts. Use --group-regex when the local naming convention differs.
    """
    stem = path.stem.strip()
    if group_regex:
        m = re.search(group_regex, stem, flags=re.IGNORECASE)
        if m:
            value = m.group(1) if m.groups() else m.group(0)
            value = normalize_token(value)
            if value:
                return value

    # Remove everything beginning with the first suffix marker. This handles
    # base_orig, base_orig_01, base_01_aug_02 and base-original3.
    marker = re.search(
        r"(?:^|[\s_.-])(?:orig(?:inal)?|aug(?:ment(?:ed)?)?)(?=$|[\s_.-]?\d|[\s_.-])",
        stem,
        flags=re.IGNORECASE,
    )
    base = stem[: marker.start()] if marker else stem
    value = normalize_token(base)
    if not value:
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        value = f"file_{digest}"
    return value


def assign_splits_by_group(rows: list[dict], ratios: tuple[float, float, float], seed: int) -> None:
    """Assign each source group to exactly one split.

    Groups are global rather than per-label because one source can contain both an
    audible target clip and a quiet clip relabeled as background. A greedy score
    approximately preserves per-class sample proportions while preventing leakage.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")

    group_rows: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        group_rows[row["group_id"]].append(i)
    if len(group_rows) < 3:
        raise ValueError("At least 3 source groups are required for train/val/test")

    labels = sorted({str(r["label"]) for r in rows})
    total = Counter(str(r["label"]) for r in rows)
    split_names = ("train", "val", "test")
    split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    target = {
        split: {label: total[label] * split_ratios[split] for label in labels}
        for split in split_names
    }
    counts = {split: Counter() for split in split_names}
    assigned: dict[str, str] = {}

    rng = random.Random(seed)
    groups = list(group_rows.items())
    rng.shuffle(groups)
    # Place large/mixed groups first because they are harder to balance later.
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    for group_id, indices in groups:
        group_count = Counter(str(rows[i]["label"]) for i in indices)
        best_split = None
        best_score = None
        for split in split_names:
            # Incremental squared-error cost. A negative value means this group
            # fills a deficit; larger train targets therefore receive groups first.
            score = 0.0
            for label in labels:
                before = counts[split][label]
                after = before + group_count[label]
                wanted = target[split][label]
                score += ((after - wanted) ** 2 - (before - wanted) ** 2) / max(1.0, wanted)
            before_total = sum(counts[split].values())
            after_total = before_total + len(indices)
            target_total = len(rows) * split_ratios[split]
            score += ((after_total - target_total) ** 2 - (before_total - target_total) ** 2) / max(1.0, target_total)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assert best_split is not None
        assigned[group_id] = best_split
        counts[best_split].update(group_count)

    for row in rows:
        row["split"] = assigned[row["group_id"]]


def validate_group_split(rows: list[dict]) -> None:
    seen: dict[str, str] = {}
    conflicts = []
    for row in rows:
        group = str(row["group_id"])
        split = str(row["split"])
        previous = seen.setdefault(group, split)
        if previous != split:
            conflicts.append((group, previous, split, row["path"]))
    if conflicts:
        preview = "\n".join(map(str, conflicts[:10]))
        raise ValueError(
            "Same-origin group appears in multiple splits. Move the whole group to one split.\n"
            + preview
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(config.DATASET_ROOT), help="Dataset root")
    parser.add_argument("--classes", default=str(config.CLASS_NAMES_PATH))
    parser.add_argument("--output", default=str(config.METADATA_CSV))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=config.TEST_RATIO)
    parser.add_argument(
        "--group-regex",
        default=config.GROUP_REGEX,
        help=(
            "Optional regex for source-recording ID in filename. The first capture group is used. "
            "Example: '(.+?)_(?:segment|chunk)_?\\d+$'"
        ),
    )
    args = parser.parse_args()
    config.validate_paths(require_dataset=True)

    root = Path(args.root).resolve()
    classes_path = Path(args.classes).resolve()
    output = Path(args.output).resolve()
    classes = read_classes(classes_path)

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.casefold() in AUDIO_EXTENSIONS)
    if not files:
        raise ValueError(f"No audio files found under {root}")

    rows: list[dict] = []
    skipped = []
    for path in files:
        rel = path.relative_to(root)
        label = detect_label(rel.parts, classes)
        if label is None:
            skipped.append(str(rel))
            continue
        split = detect_split(rel.parts)
        source_class, source_status = (label, "target_label")
        if label == "background":
            source_class, source_status = infer_background_source(rel, classes)
        rows.append(
            {
                "path": str(path),
                "label": label,
                "split": split or "",
                "source_class": source_class,
                "source_status": source_status,
                "group_id": derive_group_id(path, label, args.group_regex),
            }
        )

    if skipped:
        skipped_path = output.with_name("unrecognized_audio_paths.csv")
        pd.DataFrame({"path": skipped}).to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped)} files with no recognizable class folder: {skipped_path}")

    has_split = [bool(row["split"]) for row in rows]
    if any(has_split) and not all(has_split):
        raise ValueError("Some files have split folders and some do not. Use one consistent layout.")
    if not any(has_split):
        print(
            "WARNING: No train/val/test folders detected. Splits will be generated. "
            "The default group parser removes trailing orig/original/aug markers and their numbers. "
            "Review group_id_audit.csv before training."
        )
        assign_splits_by_group(
            rows,
            (args.train_ratio, args.val_ratio, args.test_ratio),
            args.seed,
        )

    validate_group_split(rows)
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    group_audit = (
        df.groupby("group_id")
        .agg(
            n_clips=("path", "size"),
            labels=("label", lambda x: "|".join(sorted(set(map(str, x))))),
            splits=("split", lambda x: "|".join(sorted(set(map(str, x))))),
            example_path=("path", "first"),
        )
        .reset_index()
        .sort_values(["n_clips", "group_id"], ascending=[False, True])
    )
    group_audit_path = output.with_name("group_id_audit.csv")
    group_audit.to_csv(group_audit_path, index=False)

    print(f"Wrote {len(df)} rows: {output}")
    print(f"Group audit: {group_audit_path}")
    print("label counts:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    bg = df[df["label"] == "background"]
    if not bg.empty:
        summary_path = output.with_name("background_source_summary.csv")
        bg.groupby(["source_class", "source_status"]).size().rename("count").reset_index().to_csv(
            summary_path, index=False
        )
        print(f"Background origin summary: {summary_path}")
        unresolved = bg[bg["source_class"] == "unknown"]
        if not unresolved.empty:
            unresolved_path = output.with_name("background_source_unresolved.csv")
            unresolved.to_csv(unresolved_path, index=False)
            print(f"Unresolved background origins: {len(unresolved)} -> {unresolved_path}")


if __name__ == "__main__":
    main()
