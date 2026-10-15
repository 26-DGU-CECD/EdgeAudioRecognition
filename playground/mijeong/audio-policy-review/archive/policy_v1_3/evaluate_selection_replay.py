"""Compare input hops and frozen S0/S1/S2 policies without voting or score renormalization."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from policy_core import MODEL_NAMES
from project_io import write_csv, write_json
from selection_core import decide, owned_candidates
from selection_replay_io import expected_keys, load_bundle


def evaluate(out, background_mode='all'):
    out = Path(out)
    cfg, manifest, fingerprint = load_bundle(out)
    modes = tuple(cfg['policies']) if background_mode == 'all' else (background_mode,)
    if not set(modes) <= set(cfg['policies']):
        raise ValueError('Selected background mode was not fitted in this selection run')
    canonical, outputs, costs = None, {}, []
    for model in MODEL_NAMES:
        with np.load(out / f'selection_replay_{model}.npz', allow_pickle=False) as data:
            if str(data['fingerprint'].item()) != fingerprint:
                raise ValueError(f'{model}: stale config/manifest cache')
            if (str(data['model_name'].item()) != model
                    or str(data['score_type'].item()) != 'softmax_logits'
                    or str(data['checkpoint_sha256'].item()) != manifest['checkpoint_files'][model]['sha256']):
                raise ValueError(f'{model}: wrong model or checkpoint identity')
            keys = data['keys'].astype(str)
            if keys.ndim != 1 or len(set(keys.tolist())) != len(keys):
                raise ValueError('Duplicate or malformed window keys')
            if set(keys.tolist()) != expected_keys(manifest):
                raise ValueError('Missing or extra replay windows')
            if canonical is None:
                canonical, order = keys, np.arange(len(keys))
            else:
                index = {key: i for i, key in enumerate(keys)}
                order = np.array([index[key] for key in canonical])
            z = data['logits'][order].astype(np.float64)
            if z.shape != (len(keys), len(cfg['class_names'])) or not np.isfinite(z).all():
                raise ValueError('Invalid native-head logits')
            outputs[model] = z
            parts = []
            for field in ('preprocessing_s', 'inference_s'):
                value = data[field][order].astype(np.float64)
                if value.shape != (len(keys),) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError('Invalid service timings')
                parts.append(value)
            costs.append(parts[0] + parts[1])
    costs = np.stack(costs)
    sr = int(manifest['sample_rate'])
    window_n = round(manifest['window_seconds'] * sr)
    insert_start = round(manifest['warmup_seconds'] * sr)
    insert_end = insert_start + window_n
    episode_ids = np.array([int(key.split(':')[0]) for key in canonical])
    ends = np.array([int(key.split(':')[1]) for key in canonical])
    starts = ends - window_n
    details, summary, decisions = [], [], []
    spec = cfg['selection_spec']
    for mode in modes:
        candidates = owned_candidates(outputs, cfg['class_names'], spec['expert_classes'],
                                      spec['model_class_names'], mode, spec['score_type'])
        thresholds = cfg['policies'][mode]
        for policy in ('S0', 'S1', 'S2'):
            decision = decide(candidates, policy, thresholds['global_threshold'], thresholds['class_thresholds'])
            # One trace per base-grid window; larger hops are deterministic subsets.
            for i, key in enumerate(canonical):
                decisions.append(dict(background_mode=mode, output_policy=policy, key=key,
                                      label=cfg['class_names'][int(decision['top_id'][i])],
                                      score=float(decision['score'][i]), status=str(decision['status'][i])))
            for input_policy, hop in manifest['input_policies'].items():
                hop_n = round(hop * sr)
                base_hop_n = round(manifest['base_hop_seconds'] * sr)
                if hop_n <= 0 or hop_n % base_hop_n:
                    raise ValueError('Input hops must be positive multiples of the export grid')
                selected = (ends - window_n) % hop_n == 0
                hits, first_hits, backgrounds, wrongs = [], [], [], []
                for episode in manifest['episodes']:
                    idx = np.flatnonzero(selected & (episode_ids == episode['episode_id']))
                    idx = idx[np.argsort(ends[idx])]
                    accepted = decision['accepted'][idx]
                    labels = decision['top_id'][idx]
                    relevant = (ends[idx] > insert_start) & (starts[idx] < insert_end)
                    background = episode['label'] == 'background'
                    matched = accepted & (labels == episode['label_id']) & relevant & (not background)
                    times = (ends[idx] - insert_start) / sr
                    first = float(times[matched][0]) if matched.any() else None
                    by_two = bool(np.any(matched & (times <= 2.)))
                    wrong = bool(np.any(accepted & (labels != episode['label_id'])))
                    row = dict(background_mode=mode, input_policy=input_policy, hop_seconds=hop,
                               output_policy=policy, episode_id=episode['episode_id'], label=episode['label'],
                               group_id=episode['group_id'], first_correct_audio_time_from_insert_s=first,
                               correct_by_insert_2s_audio_time=by_two if not background else None,
                               any_wrong_label=wrong,
                               background_episode_any_alert=bool(accepted.any()) if background else None,
                               preinsert_any_alert=bool(np.any(accepted & (ends[idx] <= insert_start))),
                               postinsert_nonoverlap_any_alert=bool(np.any(accepted & (starts[idx] >= insert_end))))
                    details.append(row)
                    if background:
                        backgrounds.append(bool(accepted.any()))
                    else:
                        hits.append(by_two)
                        wrongs.append(wrong)
                        if first is not None:
                            first_hits.append(first)
                summary.append(dict(
                    background_mode=mode, input_policy=input_policy, hop_seconds=hop, output_policy=policy,
                    foreground_episodes=len(hits), background_episodes=len(backgrounds),
                    foreground_hit_by_insert_2s_audio_time=float(np.mean(hits)) if hits else None,
                    foreground_missed_by_insert_2s=sum(not hit for hit in hits),
                    first_hit_audio_time_s_p50_detected_only=float(np.median(first_hits)) if first_hits else None,
                    first_hit_audio_time_s_p95_detected_only=float(np.quantile(first_hits, .95)) if first_hits else None,
                    background_episode_any_alert_rate=float(np.mean(backgrounds)) if backgrounds else None,
                    foreground_episode_any_wrong_label_rate=float(np.mean(wrongs)) if wrongs else None,
                    scheduled_windows=int(selected.sum()), scheduled_model_calls=int(selected.sum() * len(MODEL_NAMES)),
                    cached_isolated_worker_serial_service_s=float(costs[:, selected].sum()),
                    cached_step_isolated_serial_service_ms_p95=float(np.quantile(costs[:, selected].sum(axis=0), .95) * 1000),
                    experiment_id=spec['experiment_id'],
                ))
    write_csv(out / 'selection_replay_comparison.csv', summary)
    write_csv(out / 'selection_replay_per_episode.csv', details)
    write_csv(out / 'selection_replay_window_decisions.csv', decisions)
    write_json(out / 'selection_replay_scope.json', dict(
        fingerprint=fingerprint, annotation=manifest['annotation'], voting=False, score_renormalization=False,
        limitations=[
            'Thresholds were frozen from static validation clips, not calibrated for rolling windows.',
            'Insertion time is not actual event onset; these are audio-time hit proxies, not display latency.',
            'Background false alerts are counted over each whole controlled episode, including zero-only windows.',
            'Cached service sums are isolated-worker measurements, not simultaneous scheduling, queue, IPC, or UI timing.',
            'The small per-class sample is for wiring checks. No ranking or generalization claim from this smoke run.',
        ],
    ))
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--background-mode', default='all', choices=('all', 'global_max', 'exclude', 'per_model_gate'))
    args = parser.parse_args()
    summary, _ = evaluate(args.out, args.background_mode)
    for row in summary:
        print(row)
    print('Saved selection replay comparison:', Path(args.out).resolve())


if __name__ == '__main__':
    main()
