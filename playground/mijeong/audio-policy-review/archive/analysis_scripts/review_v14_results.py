import csv
import json
import hashlib
from pathlib import Path
import numpy as np

root = Path(r'C:\Users\PLASS\Desktop\audio_experiments_DEF\policy_results\V14_smoke_20261016_032646')

def rows(name):
    with (root / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

cfg = json.loads((root / 'selection_config.json').read_text(encoding='utf-8-sig'))
manifest = json.loads((root / 'replay_manifest.json').read_text(encoding='utf-8-sig'))
scope = json.loads((root / 'B_input_scope.json').read_text(encoding='utf-8-sig'))
names = cfg['class_names']
fingerprint = hashlib.sha256((root / 'selection_config.json').read_bytes() + (root / 'replay_manifest.json').read_bytes()).hexdigest()
assert fingerprint == scope['fingerprint']
result = {'source': str(root), 'split': cfg['framework']['group_split'], 'split_by_label': {}}
with np.load(root / 'selection_samples.npz', allow_pickle=False) as data:
    y, groups = data['y_true'], data['group_ids']
    sets = {}
    for split in ['fusion_fit', 'policy_fit', 'evaluation']:
        mask = data[split + '_mask']
        sets[split] = set(groups[mask].tolist())
        result['split_by_label'][split] = {
            name: {'clips': int(np.sum(mask & (y == i))), 'groups': len(set(groups[mask & (y == i)].tolist()))}
            for i, name in enumerate(names)}
    result['group_overlap'] = {a + '/' + b: len(sets[a] & sets[b]) for a, b in
                               [('fusion_fit', 'policy_fit'), ('fusion_fit', 'evaluation'), ('policy_fit', 'evaluation')]}
    eval_groups = sets['evaluation']
result['manifest'] = {'episodes': len(manifest['episodes']), 'groups': manifest['recorded_groups'],
                      'class_counts': manifest['class_counts'], 'phases': manifest['phase_offsets_seconds'],
                      'all_groups_in_evaluation': all(e['group_id'] in eval_groups for e in manifest['episodes'])}
episodes = rows('B_input_per_episode.csv')
comparison = rows('B_input_comparison.csv')
result['replay'] = []
for comparison_row in comparison:
    selected = [r for r in episodes if all(r[k] == comparison_row[k] for k in ['fusion','rule','input_policy'])]
    fg = [r for r in selected if r['label'] != 'background']
    missed = [r['label'] for r in fg if r['correct_by_clip_2s'] != 'True']
    wrong = [r['label'] for r in fg if r['any_wrong_label'] == 'True']
    prefix = [r['label'] for r in selected if r['synthetic_prefix_any_alert'] == 'True']
    assert len(fg) == int(comparison_row['foreground_episodes'])
    assert len(missed) == int(comparison_row['missed_by_clip_2s'])
    assert abs(len(wrong)/len(fg) - float(comparison_row['wrong_label_episode_rate'])) < 1e-12
    result['replay'].append({'fusion': comparison_row['fusion'], 'input': comparison_row['input_policy'],
                            'missed': missed, 'wrong_label_episodes': wrong, 'prefix_alerts': prefix,
                            'first_correct_by_label': {r['label']: r['first_correct_from_clip_s'] for r in fg},
                            'calls': int(comparison_row['scheduled_model_calls'])})
result['static_key_labels'] = [r for r in rows('framework_static_per_class.csv') if r['label'] in ['fire_alarm','knock']]
result['runtime'] = {}
for model in ['efficientat','mobilenetv4','yamnet']:
    info = json.loads((root / ('runtime_' + model + '.json')).read_text(encoding='utf-8-sig'))
    result['runtime'][model] = info
    assert info['fingerprint'] == scope['fingerprint']
result['cache_shapes'] = {}
for model in ['efficientat','mobilenetv4','yamnet']:
    with np.load(root / ('replay_' + model + '.npz'), allow_pickle=False) as data:
        result['cache_shapes'][model] = {k: list(data[k].shape) for k in data.files}
        assert data['fingerprint'].item() == fingerprint
        assert data['checkpoint_sha256'].item() == scope['checkpoint_sha256'][model]
        assert data['head_class_names'].tolist() == cfg['selection_spec']['model_class_names'][model]
        assert np.isfinite(data['logits']).all()
        sr = manifest['sample_rate']
        expected = {f"{e['episode_id']}:{end}" for e in manifest['episodes'] for end in
                    range(round(manifest['window_seconds'] * sr),
                          round((e['clip_insert_seconds'] + manifest['window_seconds'] + manifest['tail_seconds']) * sr) + 1,
                          round(manifest['base_hop_seconds'] * sr))}
        assert set(data['keys'].tolist()) == expected
result['trace_fields'] = list(rows('B_input_trace.csv')[0])
result['trace_rows'] = len(rows('B_input_trace.csv'))
assert all(float(r['result_age_s']) == 0.0 for r in rows('B_input_trace.csv'))
assert len(rows('framework_static_comparison.csv')) == 9 and len(comparison) == 9
preflight = json.loads((root / 'audio_preflight.json').read_text(encoding='utf-8-sig'))
assert preflight['passed'] and preflight['failed_files'] == 0
assert preflight['manifest_sha256'] == hashlib.sha256((root / 'replay_manifest.json').read_bytes()).hexdigest()
result['integrity_checks'] = 'passed: config/manifest fingerprint, cache checkpoint hashes, head order, finite logits, exact window keys, episode summary reconciliation, zero audio-clock ages, preflight manifest hash'
out = Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_smoke_review')
out.mkdir(parents=True, exist_ok=True)
(out / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ['integrity_checks','group_overlap','manifest','trace_rows']}, ensure_ascii=False, indent=2))
guide = out / '結果検討.md'
report = out / '결과검토_다음실험.md'
assert report.exists() and report.read_text(encoding='utf-8').count('```') % 2 == 0
print('Report verified:', report)
