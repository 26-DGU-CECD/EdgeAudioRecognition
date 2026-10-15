"""Shared provenance and path checks for no-voting controlled replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from policy_core import MODEL_NAMES
from selection_core import BACKGROUND_MODES, decide, owned_candidates

SOURCE_NAMES = (
    'class_names.txt', 'common_audio.py', 'common_torch.py', 'common_metadata.py',
    'evaluation_utils.py', 'config.py', 'train_efficientat.py', 'train_mobilenetv4.py',
    'third_party/EfficientAT/models/mn/model.py', 'third_party/EfficientAT/models/mn/utils.py',
    'third_party/EfficientAT/models/mn/block_types.py',
    'third_party/EfficientAT/models/mn/attention_pooling.py',
    'third_party/EfficientAT/helpers/utils.py',
    'third_party/EfficientAT/metadata/class_labels_indices.csv',
)

REPLAY_SOURCE_NAMES = (
    'policy_core.py', 'selection_core.py', 'project_io.py', 'selection_replay_io.py',
    'prepare_selection_replay.py', 'export_selection_replay.py', 'evaluate_selection_replay.py',
    'check_selection_reference.py',
)


def replay_source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha_file(root / name) for name in REPLAY_SOURCE_NAMES}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_calibration_group(seed, group):
    return int(hashlib.sha256(f'{seed}|{group}'.encode()).hexdigest()[:16], 16) % 2 == 0


def map_audio_path(original, old_root, local_root):
    # Metadata identities use POSIX paths, even when executed on Windows.
    original, old_root = PurePosixPath(original), PurePosixPath(old_root)
    if '..' in original.parts:
        raise ValueError('Parent traversal in metadata path')
    relative = original.relative_to(old_root)
    if not relative.parts:
        raise ValueError('Expected an audio file below the dataset root')
    root = Path(local_root).resolve()
    mapped = root.joinpath(*relative.parts).resolve()
    if not mapped.is_relative_to(root):
        raise ValueError('Mapped audio escaped the dataset root')
    return mapped


def validate_selection(config):
    if (config.get('schema') != 'owned_max_selection_v1' or config.get('voting') is not False
            or config.get('score_renormalization') is not False):
        raise ValueError('Expected a v1.2 no-voting selection configuration')
    spec, names = config['selection_spec'], config['class_names']
    if spec['score_type'] != 'softmax_logits':
        raise ValueError('The provided runtime adapter requires softmax_logits, not raw-logit max or probabilities')
    if set(spec['model_class_names']) != set(MODEL_NAMES):
        raise ValueError('This runtime adapter requires the three provided model identities')
    if any(spec['model_class_names'][m] != names for m in MODEL_NAMES):
        raise ValueError('This runtime adapter supports the provided full heads only; custom subset-head loading needs an adapter')
    import numpy as np
    candidates = owned_candidates({m: np.zeros((1, len(names))) for m in MODEL_NAMES},
                                  names, spec['expert_classes'], spec['model_class_names'], 'global_max')
    if not config['policies'] or not set(config['policies']) <= set(BACKGROUND_MODES):
        raise ValueError('Invalid background modes')
    for policy in config['policies'].values():
        decide(candidates, 'S1', policy['global_threshold'])
        decide(candidates, 'S2', class_thresholds=policy['class_thresholds'])


def load_bundle(out):
    out = Path(out)
    config_bytes = (out / 'selection_config.json').read_bytes()
    manifest_bytes = (out / 'selection_replay_manifest.json').read_bytes()
    config, manifest = json.loads(config_bytes), json.loads(manifest_bytes)
    validate_selection(config)
    if manifest.get('schema') != 'owned_max_replay_v1':
        raise ValueError('Not a selection replay manifest')
    if manifest.get('replay_sources') != replay_source_hashes():
        raise ValueError('Replay implementation changed; prepare a new output folder')
    if manifest['selection_config_sha256'] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError('Frozen selection configuration changed; prepare a new replay folder')
    if manifest['class_names'] != config['class_names'] or not manifest['episodes']:
        raise ValueError('Manifest class names or episode list invalid')
    episodes = manifest['episodes']
    if [e['episode_id'] for e in episodes] != list(range(len(episodes))):
        raise ValueError('Episode identities must be unique and consecutive')
    if len({e['group_id'] for e in episodes}) != len(episodes):
        raise ValueError('A recorded group was reused within this replay')
    for episode in episodes:
        if (config['class_names'][episode['label_id']] != episode['label']
                or is_calibration_group(config['seed'], episode['group_id'])):
            raise ValueError('Wrong label identity or calibration group entered replay')
    fingerprint = hashlib.sha256(config_bytes + manifest_bytes).hexdigest()
    return config, manifest, fingerprint


def validate_project_sources(project_root, manifest):
    project_root = Path(project_root)
    for relative, expected in manifest['project_sources'].items():
        if sha_file(project_root / relative) != expected:
            raise ValueError(f'Project source changed: {relative}; use a new prepared replay')


def verify_checkpoint(manifest, model):
    item = manifest['checkpoint_files'][model]
    path = Path(item['path'])
    if sha_file(path) != item['sha256']:
        raise ValueError(f'Checkpoint changed: {model}')
    return path


def verify_audio(episode):
    path = Path(episode['path'])
    if sha_file(path) != episode['audio_sha256']:
        raise ValueError(f"Source audio changed: {episode['path']}")
    return path


def expected_keys(manifest):
    sr = manifest['sample_rate']
    window_n = round(manifest['window_seconds'] * sr)
    hop_n = round(manifest['base_hop_seconds'] * sr)
    total = round((manifest['warmup_seconds'] + manifest['window_seconds'] + manifest['tail_seconds']) * sr)
    return {f"{e['episode_id']}:{end}" for e in manifest['episodes']
            for end in range(window_n, total + 1, hop_n)}
