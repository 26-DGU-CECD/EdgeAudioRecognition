"""Read-only comparison of original prediction identities and local audio files."""
import csv
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

PROJECT = Path(r'C:\Users\PLASS\Desktop\audio_experiments_DEF')
ROOTS = [Path(r'C:\Users\PLASS\Desktop\alarm_aug_dataset'), Path(r'D:\Sound_Dataset_aug')]
OUT = Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs')
META = PROJECT / 'metadata.csv'
ORIGINAL_ROOT = PurePosixPath('/home/plass/YAMNet/aug_data')
EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aif', '.aiff'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header(path):
    try:
        # stdlib wave rejects IEEE_FLOAT (format 3), which is valid WAV.
        # Parse RIFF headers and inspect up to the first two seconds of samples.
        with path.open('rb') as stream:
            signature = stream.read(12)
            if signature[:4] != b'RIFF' or signature[8:12] != b'WAVE':
                raise ValueError('Not a supported little-endian RIFF/WAVE file')
            fmt = None
            while True:
                chunk = stream.read(8)
                if len(chunk) != 8:
                    raise ValueError('No data chunk found')
                name, size = struct.unpack('<4sI', chunk)
                if name == b'fmt ':
                    fmt = stream.read(size)
                    if size % 2:
                        stream.seek(1, 1)
                elif name == b'data':
                    if fmt is None or len(fmt) < 16:
                        raise ValueError('Missing or invalid fmt chunk')
                    code, channels, sample_rate, byte_rate, align, bits = struct.unpack('<HHIIHH', fmt[:16])
                    if code == 0xFFFE and len(fmt) >= 40:
                        code = struct.unpack('<H', fmt[24:26])[0]
                    if channels <= 0 or sample_rate <= 0 or align <= 0 or size % align:
                        raise ValueError('Invalid dimensions or data chunk size')
                    result = dict(status='wav_header_read', format_code=code, channels=channels,
                                  sample_rate=sample_rate, frames=size // align,
                                  duration_seconds=(size // align) / sample_rate,
                                  bits_per_sample=bits)
                    sample_bytes = stream.read(min(size, sample_rate * align * 2))
                    if len(sample_bytes) != min(size, sample_rate * align * 2):
                        raise ValueError('Truncated sample data')
                    dtype = {(3, 32): '<f4', (3, 64): '<f8', (1, 16): '<i2', (1, 32): '<i4'}.get((code, bits))
                    if dtype is not None:
                        samples = np.frombuffer(sample_bytes, dtype=dtype).astype(np.float64)
                        if code == 1:
                            samples /= 2 ** (bits - 1)
                        result.update(status='wav_header_and_first_2s_samples_read',
                                      sampled_frames=len(samples) // channels,
                                      samples_all_finite=bool(np.isfinite(samples).all()),
                                      sample_abs_peak=float(np.max(np.abs(samples))) if len(samples) else None)
                    return result
                else:
                    stream.seek(size + size % 2, 1)
    except Exception as error:
        return dict(status='wav_inspection_failed; decoder_check_needed',
                    error_type=type(error).__name__, error=str(error))


original_hash = sha(META)
with META.open(encoding='utf-8-sig', newline='') as stream:
    metadata = list(csv.DictReader(stream))
names = (PROJECT / 'class_names.txt').read_text(encoding='utf-8-sig').splitlines()
for row in metadata:
    row['_relative'] = PurePosixPath(row['path']).relative_to(ORIGINAL_ROOT).as_posix()
assert len({row['path'] for row in metadata}) == len(metadata)
print('Metadata records:', len(metadata), flush=True)

results = []
for root in ROOTS:
    print('Indexing:', root, flush=True)
    index = set()
    basename_paths = defaultdict(list)
    samples = defaultdict(list)
    folder_counts = Counter()
    walk_errors = []
    for parent, directories, filenames in os.walk(root, onerror=lambda error: walk_errors.append(str(error))):
        for filename in filenames:
            if Path(filename).suffix.lower() not in EXTENSIONS:
                continue
            path = Path(parent) / filename
            relative = path.relative_to(root).as_posix()
            index.add(relative)
            basename_paths[filename].append(relative)
            top_folder = relative.split('/')[0] if '/' in relative else '(root)'
            folder_counts[top_folder] += 1
            # Header sampling is deterministic within each traversal by taking
            # the lexicographically first two actual files per top folder.
            samples[top_folder] = sorted(samples[top_folder] + [relative])[:2]
    print('Indexed audio files:', len(index), flush=True)
    by_label = {}
    matched = 0
    exact_basename_unique = 0
    exact_basename_ambiguous = 0
    missing_examples = []
    basename_examples = []
    for name in names:
        rows = [row for row in metadata if row['label'] == name]
        relative_match = sum(row['_relative'] in index for row in rows)
        unique_match = sum(len(basename_paths.get(PurePosixPath(row['path']).name, [])) == 1 for row in rows)
        ambiguous = sum(len(basename_paths.get(PurePosixPath(row['path']).name, [])) > 1 for row in rows)
        by_label[name] = dict(metadata_clips=len(rows), relative_path_matches=relative_match,
                             unique_basename_matches=unique_match, ambiguous_basename_matches=ambiguous)
        matched += relative_match
        exact_basename_unique += unique_match
        exact_basename_ambiguous += ambiguous
    for row in metadata:
        if row['_relative'] not in index and len(missing_examples) < 5:
            missing_examples.append(dict(label=row['label'], original_path=row['path'],
                                         expected_local_path=str(root / row['_relative'])))
        options = basename_paths.get(PurePosixPath(row['path']).name, [])
        if row['_relative'] not in index and len(options) == 1 and len(basename_examples) < 5:
            basename_examples.append(dict(original_label=row['label'], actual_relative_path=options[0]))
    declared = None
    label_file = root / 'dataset_labels.csv'
    if label_file.is_file():
        with label_file.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        declared = dict(columns=list(rows[0]) if rows else [], rows=len(rows),
                        label_counts=dict(Counter(row.get('label', '') for row in rows)),
                        declared_audio_paths_found=sum(row.get('file_path', '').replace('\\', '/') in index for row in rows),
                        sha256=sha(label_file))
    headers = []
    for folder in sorted(samples):
        for relative in samples[folder]:
            headers.append(dict(relative_path=relative, **header(root / relative)))
    result = dict(root=str(root), audio_file_count=len(index), top_folder_audio_counts=dict(folder_counts),
                  traversal_errors=walk_errors, metadata_clips=len(metadata),
                  relative_path_matches=matched, unique_basename_matches=exact_basename_unique,
                  ambiguous_basename_matches=exact_basename_ambiguous,
                  by_label=by_label, missing_examples=missing_examples,
                  filename_only_mapping_examples=basename_examples,
                  dataset_labels=declared, header_samples=headers,
                  header_scope='At most two lexicographically first audio files per top folder. RIFF header plus first two seconds of supported PCM/float samples; no full-file decode or model preprocessing.')
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('header_samples', 'missing_examples', 'by_label')}, ensure_ascii=False), flush=True)
    print('Header sample summary:', dict(Counter((r['status'], r.get('sample_rate'), r.get('channels')) for r in headers)), flush=True)

assert sha(META) == original_hash
report = dict(project_root=str(PROJECT), metadata_path=str(META), metadata_sha256=original_hash,
              old_dataset_root=str(ORIGINAL_ROOT), metadata_rows=len(metadata),
              split_counts=dict(Counter(row['split'] for row in metadata)), class_names=names,
              original_metadata_unchanged=True, roots=results,
              limitations=['Matching paths or filenames do not establish content identity with the historical training audio.',
                           'No relabeling, copying, training, new model inference, or checkpoint changes were performed.',
                           'Sampled files do not establish full-dataset decode health or event onset annotations.'])
OUT.mkdir(parents=True, exist_ok=True)
output = OUT / 'dataset_connection_audit.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved:', output, flush=True)
