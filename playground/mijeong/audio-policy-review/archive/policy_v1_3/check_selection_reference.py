"""Compare each replay's full-source 2s window with the original saved validation prediction."""
import argparse
from pathlib import Path

import numpy as np

from policy_core import MODEL_NAMES, softmax
from project_io import write_json
from selection_replay_io import load_bundle, sha_file


def check(out, probability_atol=.001):
    if not 0 <= probability_atol <= 1:
        raise ValueError('probability-atol must be in [0,1]')
    out = Path(out)
    config, manifest, fingerprint = load_bundle(out)
    full_end = round((manifest['warmup_seconds'] + manifest['window_seconds']) * manifest['sample_rate'])
    results = []
    for model in MODEL_NAMES:
        item = config['prediction_provenance'][model]
        if sha_file(item['path']) != item['sha256']:
            raise ValueError('Original saved prediction file changed')
        with np.load(item['path'], allow_pickle=False) as data:
            original = {p: z for p, z in zip(data['paths'].astype(str), data['logits'])}
        with np.load(out / f'selection_replay_{model}.npz', allow_pickle=False) as data:
            if (str(data['fingerprint'].item()) != fingerprint
                    or str(data['checkpoint_sha256'].item()) != manifest['checkpoint_files'][model]['sha256']
                    or str(data['model_name'].item()) != model
                    or str(data['score_type'].item()) != 'softmax_logits'):
                raise ValueError('Wrong model identity or stale replay cache')
            keys = data['keys'].astype(str)
            if len(set(keys.tolist())) != len(keys):
                raise ValueError('Duplicate replay keys')
            replay = dict(zip(keys, data['logits']))
        for episode in manifest['episodes']:
            old = np.asarray(original[episode['original_path']], dtype=np.float64)
            new = np.asarray(replay[f"{episode['episode_id']}:{full_end}"], dtype=np.float64)
            if (old.shape != (len(config['class_names']),) or new.shape != old.shape
                    or not np.isfinite(new).all() or not np.isfinite(old).all()):
                raise ValueError('Malformed full-clip reference scores')
            difference = float(np.abs(softmax(old) - softmax(new)).max())
            top_match = bool(old.argmax() == new.argmax())
            results.append(dict(model=model, label=episode['label'], episode_id=episode['episode_id'],
                                original_path=episode['original_path'], max_probability_abs_diff=difference,
                                max_logit_abs_diff=float(np.abs(old - new).max()), top1_matches=top_match,
                                within_probability_tolerance=difference <= probability_atol))
    report = dict(fingerprint=fingerprint, probability_atol=probability_atol,
                  passed=all(r['top1_matches'] and r['within_probability_tolerance'] for r in results),
                  checked_model_clip_pairs=len(results), results=results,
                  scope='Sampled full-clip model/preprocessing compatibility only; .001 is a diagnostic tolerance, not a correctness guarantee.')
    write_json(out / 'selection_reference_check.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--probability-atol', type=float, default=.001)
    args = parser.parse_args()
    report = check(args.out, args.probability_atol)
    print('Full-clip reference compatibility:', 'PASS' if report['passed'] else 'REVIEW REQUIRED')
    print('Checked model/clip pairs:', report['checked_model_clip_pairs'])
    if not report['passed']:
        print('Inspect selection_reference_check.json before interpreting the replay policies.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
