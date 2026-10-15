import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

SOURCE = Path(r'C:\Users\PLASS\Desktop\audio_experiments_DEF\policy_results\V14_pilot_20261016_034601')
CODE = Path(r'C:\Users\PLASS\Desktop\audio-policy-review-20261015T164106Z-1-001\audio-policy-review\audio_policy_pilot_v1_4')
DELIVERY = Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_expanded_next')
EXPERIMENT = DELIVERY / 'experiment'
WORK = Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\work\v14_advance_logs')
for folder in [DELIVERY, EXPERIMENT, WORK]:
    folder.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def key(row):
    return tuple(row[k] for k in ['fusion','rule','input_policy'])

def flag(value):
    return value is True or value == 'True'

def rate(rows, name):
    return sum(flag(r[name]) for r in rows) / len(rows) if rows else None

manifest = json.loads((SOURCE / 'replay_manifest.json').read_text(encoding='utf-8-sig'))
config = json.loads((SOURCE / 'selection_config.json').read_text(encoding='utf-8-sig'))
fingerprint = hashlib.sha256((SOURCE/'selection_config.json').read_bytes() + (SOURCE/'replay_manifest.json').read_bytes()).hexdigest()
for tag in ['B_expanded','B_expanded_fifo']:
    scope = json.loads((SOURCE/(tag+'_scope.json')).read_text(encoding='utf-8-sig'))
    assert scope['fingerprint'] == fingerprint

copy_names = ['selection_config.json','selection_samples.npz','replay_manifest.json','audio_preflight.json','audio_preflight.csv']
for model in ['efficientat','mobilenetv4','yamnet']:
    copy_names += [f'replay_{model}.npz',f'runtime_{model}.json']
    with np.load(SOURCE/f'replay_{model}.npz', allow_pickle=False) as cache:
        assert cache['fingerprint'].item() == fingerprint
        assert cache['checkpoint_sha256'].item() == scope['checkpoint_sha256'][model]
        assert np.isfinite(cache['logits']).all()

hashes = {}
for name in copy_names:
    hashes[name] = hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
    target = EXPERIMENT/name
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[name], 'Refusing changed input copy'
    else:
        shutil.copy2(SOURCE/name,target)
code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')}
provenance = dict(source=str(SOURCE), source_code=str(CODE), python=sys.version, fingerprint=fingerprint,
                  copied_input_sha256=hashes, code_sha256=code_hashes,
                  inference_executed=False, source_results_modified=False)
(DELIVERY/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

audio = read_csv(SOURCE/'B_expanded_comparison.csv')
fifo = read_csv(SOURCE/'B_expanded_fifo_comparison.csv')
episodes_audio = read_csv(SOURCE/'B_expanded_per_episode.csv')
episodes_fifo = read_csv(SOURCE/'B_expanded_fifo_per_episode.csv')
assert len(audio)==len(fifo)==27
assert len(episodes_audio)==len(episodes_fifo)==27*len(manifest['episodes'])
detail_map = {(key(r),r['episode_id']):r for r in episodes_fifo}
paired = []
phase_class = []
for a,b in zip(audio,fifo):
    assert key(a)==key(b)
    subset = [r for r in episodes_audio if key(r)==key(a)]
    fg = [r for r in subset if r['label']!='background']
    bg = [r for r in subset if r['label']=='background']
    matched_fifo = [detail_map[(key(r),r['episode_id'])] for r in fg]
    assert len(fg)==416 and len(bg)==32
    assert abs(rate(fg,'correct_by_clip_2s')-float(a['hit_by_clip_2s']))<1e-12
    assert abs(rate(matched_fifo,'correct_by_clip_2s')-float(b['hit_by_clip_2s']))<1e-12
    assert abs(rate(fg,'any_wrong_label')-float(b['wrong_label_episode_rate']))<1e-12
    lost = [r for r in fg if flag(r['correct_by_clip_2s']) and not flag(detail_map[(key(r),r['episode_id'])]['correct_by_clip_2s'])]
    assert all(float(r['first_correct_from_clip_s'])==2.0 for r in lost)
    delays = [float(detail_map[(key(r),r['episode_id'])]['first_correct_from_clip_s'])-float(r['first_correct_from_clip_s']) for r in fg if r['first_correct_from_clip_s']]
    paired.append(dict(fusion=a['fusion'],rule=a['rule'],input_policy=a['input_policy'],
                       audio_hit=float(a['hit_by_clip_2s']),fifo_hit=float(b['hit_by_clip_2s']),
                       boundary_losses=len(lost), median_added_ms=float(np.median(delays))*1000,
                       max_added_ms=float(max(delays))*1000,
                       wrong_label=float(b['wrong_label_episode_rate']),background_alert=float(b['background_episode_alert_rate']),
                       scheduled_model_calls=int(b['scheduled_model_calls'])))
    for field in ['phase_offset_seconds','label']:
        for value in sorted({r[field] for r in subset}):
            local = [r for r in subset if r[field]==value]
            local_fg = [r for r in local if r['label']!='background']
            local_fifo = [detail_map[(key(r),r['episode_id'])] for r in local_fg]
            phase_class.append(dict(fusion=a['fusion'],rule=a['rule'],input_policy=a['input_policy'],
                                    breakdown=field,value=value,foreground_episodes=len(local_fg),
                                    audio_hit=rate(local_fg,'correct_by_clip_2s'),fifo_hit=rate(local_fifo,'correct_by_clip_2s'),
                                    wrong_label=rate(local_fg,'any_wrong_label'),
                                    prefix_alerts=sum(flag(r['synthetic_prefix_any_alert']) for r in local)))
save_csv(DELIVERY/'baseline_paired_summary.csv',paired)
save_csv(DELIVERY/'baseline_phase_class_summary.csv',phase_class)
print('BASELINE', json.dumps(paired,ensure_ascii=False),flush=True)

def run(tag, fusions, rules, inputs, temporal, timing, async_policy='wait_all'):
    cmd = [sys.executable,'-B',str(CODE/'evaluate_framework_replay.py'),'--out',str(EXPERIMENT),
           '--fusions',*fusions,'--rules',*rules,'--inputs',*inputs,'--temporal',*temporal,
           '--timing',timing,'--async-policy',async_policy,'--result-tag',tag]
    started=time.time()
    proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
    (WORK/(tag+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'{tag} failed: '+proc.stderr[-5000:])
    rows=read_csv(EXPERIMENT/(tag+'_comparison.csv'))
    print('FINISHED',tag,'rows',len(rows),'seconds',round(time.time()-started,2),flush=True)
    for row in rows:
        if tag.startswith('B_') and row['input_policy']=='I3':
            print('ADAPTIVE',json.dumps(row,ensure_ascii=False),flush=True)
    return rows

if __name__=='__main__':
    if '--followup' in sys.argv:
        for timing,suffix in [('audio_clock','audio'),('independent_worker_sim','fifo')]:
            run('C_temporal_'+suffix,['F1','F2'],['R0','R2'],['I2','I3'],['T0','T1','T2'],timing)
        for async_policy in ['wait_all','bounded']:
            run('D_'+async_policy,['F1','F2'],['R0','R2'],['I2'],['T0'],'independent_worker_sim',async_policy)
    else:
        for timing,suffix in [('audio_clock','audio'),('independent_worker_sim','fifo')]:
            run('B_adaptive_'+suffix,['F1','F2'],['R0','R2'],['I1','I2','I3'],['T0'],timing)
