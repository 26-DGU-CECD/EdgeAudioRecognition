"""Prepare one local audio clip per recorded evaluation group, without model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from policy_core import MODEL_NAMES
from project_io import ProjectFiles, write_json
from selection_replay_io import (
    SOURCE_NAMES, is_calibration_group, map_audio_path, replay_source_hashes, sha_file, validate_selection,
)


def prepare(project_root, selection_config, dataset_root, out, max_per_class=1,
            old_dataset_root='/home/plass/YAMNet/aug_data'):
    if max_per_class < 1:
        raise ValueError('max-per-class must be positive')
    project = ProjectFiles(project_root)
    dataset_root, out = Path(dataset_root).resolve(), Path(out).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new empty output folder: {out}')
    config_bytes = Path(selection_config).read_bytes()
    cfg = json.loads(config_bytes)
    validate_selection(cfg)
    if project.class_names != cfg['class_names']:
        raise ValueError('Project class order differs from the frozen selection run')
    metadata_file = project.locate('metadata.csv')
    if sha_file(metadata_file) != cfg['metadata_sha256']:
        raise ValueError('Metadata changed since selection fitting')
    with metadata_file.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == 'val']
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate validation metadata paths')
    row_by_path = {r['path']: r for r in rows}
    checkpoints = {}
    for model in MODEL_NAMES:
        item = cfg['prediction_provenance'][model]
        prediction_file = Path(item['path'])
        if sha_file(prediction_file) != item['sha256']:
            raise ValueError(f'Saved predictions changed: {model}')
        with np.load(prediction_file, allow_pickle=False) as data:
            paths = data['paths'].astype(str)
            targets = data['y_true'].astype(np.int64)
            if data['logits'].shape != (len(paths), len(cfg['class_names'])):
                raise ValueError('Unexpected saved prediction head size')
        if len(set(paths.tolist())) != len(paths) or set(paths.tolist()) != set(row_by_path):
            raise ValueError('Saved prediction samples differ from validation metadata')
        if targets.shape != paths.shape or (targets < 0).any() or (targets >= len(cfg['class_names'])).any():
            raise ValueError('Invalid global prediction target IDs')
        if any(cfg['class_names'][int(y)] != row_by_path[p]['label'] for p, y in zip(paths, targets)):
            raise ValueError('Global prediction targets do not match metadata')
        suffix = 'best.keras' if model == 'yamnet' else 'best.pt'
        checkpoint = prediction_file.parent / suffix
        if not checkpoint.is_file():
            raise FileNotFoundError(f'No checkpoint next to saved predictions: {checkpoint}')
        checkpoints[model] = dict(path=str(checkpoint.resolve()), sha256=sha_file(checkpoint))
    calibration = [r for r in rows if is_calibration_group(cfg['seed'], r['group_id'])]
    evaluation = [r for r in rows if not is_calibration_group(cfg['seed'], r['group_id'])]
    if (len(calibration) != cfg['calibration_clips'] or len(evaluation) != cfg['evaluation_clips']
            or len({r['group_id'] for r in calibration}) != cfg['calibration_groups']
            or len({r['group_id'] for r in evaluation}) != cfg['evaluation_groups']):
        raise ValueError('Recomputed split differs from selection configuration')
    episodes, used, missing = [], set(), []
    for label_id, label in enumerate(cfg['class_names']):
        candidates = [r for r in evaluation if r['label'] == label]
        candidates.sort(key=lambda r: (hashlib.sha256(f"{cfg['seed']}|{r['group_id']}|{r['path']}".encode()).hexdigest(), r['path']))
        kept = 0
        for row in candidates:
            if row['group_id'] in used:
                continue
            path = map_audio_path(row['path'], old_dataset_root, dataset_root)
            if not path.is_file():
                missing.append(str(path))
                continue
            used.add(row['group_id'])
            episodes.append(dict(episode_id=len(episodes), path=str(path),
                                 original_path=row['path'], label=label, label_id=label_id,
                                 group_id=row['group_id'], audio_sha256=sha_file(path)))
            kept += 1
            if kept >= max_per_class:
                break
        if not kept:
            raise FileNotFoundError(f'No accessible evaluation audio for {label}; examples={missing[-3:]}')
    manifest = dict(
        schema='owned_max_replay_v1', project_root=str(project.root),
        selection_config_source=str(Path(selection_config).resolve()),
        selection_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        class_names=cfg['class_names'], dataset_root=str(dataset_root), old_dataset_root=old_dataset_root,
        checkpoint_files=checkpoints,
        replay_sources=replay_source_hashes(),
        project_sources={relative: sha_file(project.locate(relative)) for relative in SOURCE_NAMES},
        sample_rate=16000, window_seconds=2., base_hop_seconds=.5,
        warmup_seconds=2., tail_seconds=2., input_policies={'I0': 2., 'I1': 1., 'I2': .5},
        audio_config=dict(highpass_hz=80., highpass_order=2, silence_frame_ms=100.,
                          silence_threshold_dbfs=-30., target_rms_dbfs=-20., max_normalize_gain_db=20.),
        selection='One clip per recorded group, only the frozen selection run evaluation groups.',
        annotation='Controlled 2s zeros + first 2s of source clip + 2s zeros; insertion-aligned smoke replay, not real event onset, natural background, or UI latency.',
        requested_max_per_class=max_per_class, class_counts=dict(Counter(e['label'] for e in episodes)),
        missing_audio_count=len(missing), episodes=episodes,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'selection_config.json').write_bytes(config_bytes)
    write_json(out / 'selection_replay_manifest.json', manifest)
    write_json(out / 'selection_replay_preparation.json', dict(
        episodes=len(episodes), class_counts=manifest['class_counts'],
        original_metadata_sha256=cfg['metadata_sha256'],
        checkpoint_files_exist=True, calibration_group_overlap=0,
        scope='Preparation only; audio hashes checked, no preprocessing or model inference executed.',
    ))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--selection-config', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--old-dataset-root', default='/home/plass/YAMNet/aug_data')
    parser.add_argument('--max-per-class', type=int, default=1)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest = prepare(**vars(args))
    print('Prepared episodes:', len(manifest['episodes']))
    print('Class counts:', manifest['class_counts'])
    print('Saved:', Path(args.out).resolve())


if __name__ == '__main__':
    main()
