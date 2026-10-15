import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_expanded_next\experiment')
OUT=Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_experiment_analysis')
OUT.mkdir(parents=True,exist_ok=True)
TAGS=['B_adaptive_audio','B_adaptive_fifo','C_temporal_audio','C_temporal_fifo','D_wait_all','D_bounded']

def read(tag,suffix):
    with (ROOT/f'{tag}_{suffix}.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def yes(x): return x=='True'
def setting(r): return tuple(r[k] for k in ['fusion','rule','input_policy','temporal'])
def erow(r): return (*setting(r),int(r['episode_id']))
def trkey(r): return (*erow(r),float(r['audio_end_s']))
def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

manifest=json.loads((ROOT/'replay_manifest.json').read_text(encoding='utf-8-sig'))
cfg=json.loads((ROOT/'selection_config.json').read_text(encoding='utf-8-sig'))
names=cfg['class_names']
episodes={e['episode_id']:e for e in manifest['episodes']}
fingerprint=hashlib.sha256((ROOT/'selection_config.json').read_bytes()+(ROOT/'replay_manifest.json').read_bytes()).hexdigest()
summary={}; details={}
for tag in TAGS:
    summary[tag]=read(tag,'comparison');details[tag]=read(tag,'per_episode')
    assert len(details[tag])==len(summary[tag])*len(episodes)
    scope=json.loads((ROOT/f'{tag}_scope.json').read_text(encoding='utf-8-sig'))
    assert scope['fingerprint']==fingerprint
    grouped=defaultdict(list)
    for row in details[tag]: grouped[setting(row)].append(row)
    for s in summary[tag]:
        local=grouped[setting(s)];fg=[r for r in local if r['label']!='background'];bg=[r for r in local if r['label']=='background']
        assert len(fg)==416 and len(bg)==32
        for metric,field,denom in [('hit_by_clip_2s','correct_by_clip_2s',fg),('wrong_label_episode_rate','any_wrong_label',fg),('background_episode_alert_rate','background_any_alert',bg)]:
            assert abs(float(s[metric])-sum(yes(r[field]) for r in denom)/len(denom))<1e-12
        assert int(s['scheduled_model_calls'])==sum(int(r['scheduled_model_calls']) for r in local)

class_rows=[]
for tag in ['B_adaptive_fifo','C_temporal_fifo']:
    grouped=defaultdict(list)
    for r in details[tag]: grouped[(*setting(r),r['label'])].append(r)
    for key,local in grouped.items():
        fg=key[-1]!='background'
        times=[float(r['first_correct_from_clip_s']) for r in local if r['first_correct_from_clip_s']]
        class_rows.append(dict(stage=tag,fusion=key[0],rule=key[1],input_policy=key[2],temporal=key[3],label=key[4],
                               groups=len({r['group_id'] for r in local}),episodes=len(local),
                               hit_by_2s=sum(yes(r['correct_by_clip_2s']) for r in local) if fg else '',
                               late_detection=sum(bool(r['first_correct_from_clip_s']) and float(r['first_correct_from_clip_s'])>2 for r in local) if fg else '',
                               never_detected=sum(not r['first_correct_from_clip_s'] for r in local) if fg else '',
                               wrong_episodes=sum(yes(r['any_wrong_label']) for r in local) if fg else '',
                               background_alerts=sum(yes(r['background_any_alert']) for r in local) if not fg else '',
                               first_correct_p50=float(np.median(times)) if times else ''))
save('라벨별_검출과_오표시.csv',class_rows)

paired=[]
for tag in ['B_adaptive_audio','B_adaptive_fifo']:
    index={erow(r):r for r in details[tag]}
    for fusion in ['F1','F2']:
        for rule in ['R0','R2']:
            a=[r for r in details[tag] if setting(r)==(fusion,rule,'I2','T0') and r['label']!='background']
            b=[index[(fusion,rule,'I3','T0',int(r['episode_id']))] for r in a]
            lost=[r for r,z in zip(a,b) if yes(r['correct_by_clip_2s']) and not yes(z['correct_by_clip_2s'])]
            gained=[z for r,z in zip(a,b) if not yes(r['correct_by_clip_2s']) and yes(z['correct_by_clip_2s'])]
            paired.append(dict(stage=tag,fusion=fusion,rule=rule,lost=len(lost),gained=len(gained),
                               lost_group_count=len({r['group_id'] for r in lost}),
                               lost_labels=dict(Counter(r['label'] for r in lost)),
                               gained_labels=dict(Counter(r['label'] for r in gained))))

trace=read('B_adaptive_fifo','trace')
region_counts=defaultdict(Counter)
for r in trace:
    if r['fusion']!='F1' or r['rule']!='R0':continue
    e=episodes[int(r['episode_id'])];end=float(r['audio_end_s']);insert=e['clip_insert_seconds']
    region='synthetic_prefix' if end<=insert else ('source_overlap' if end-2<insert+2 else 'synthetic_tail')
    region_counts[r['input_policy']][region]+=1
confusion=defaultdict(lambda:dict(episode=set(),group=set(),windows=0))
for r in trace:
    if setting(r)!=('F1','R0','I3','T0') or not r['displayed_id']: continue
    e=episodes[int(r['episode_id'])];end=float(r['audio_end_s']);insert=e['clip_insert_seconds']
    if not(end>insert and end-2<insert+2):continue
    displayed=names[int(r['displayed_id'])]
    if displayed==e['label']:continue
    c=confusion[(e['label'],displayed)];c['episode'].add(e['episode_id']);c['group'].add(e['group_id']);c['windows']+=1
confusion_rows=[dict(true_label=a,displayed_label=b,groups=len(c['group']),episodes=len(c['episode']),windows=c['windows']) for (a,b),c in confusion.items()]
confusion_rows.sort(key=lambda r:(-r['groups'],-r['episodes'],r['true_label']))
save('F1_R0_I3_오표시_상세.csv',confusion_rows)
case_rows=[]
for (true,predicted),c in confusion.items():
    for group in sorted(c['group']):
        matches=[e for e in episodes.values() if e['group_id']==group]
        local_ids={e['episode_id'] for e in matches}&c['episode']
        case_rows.append(dict(true_label=true,displayed_label=predicted,group_id=group,
                              wrong_phase_episodes=len(local_ids),audio_path=matches[0]['path']))
save('F1_R0_I3_오표시_원음목록.csv',case_rows)

ta=read('C_temporal_audio','trace');tf=read('C_temporal_fifo','trace')
audio_index={trkey(r):r for r in ta}
state={};near=[];different=[]
for r in tf:
    if r['temporal']!='T2': continue
    k=erow(r);now=float(r['decision_time_s']);candidate=int(r['raw_top_id']) if yes(r['accepted']) else None
    pending,since,count=state.get(k,(None,None,0))
    if candidate is None: pending,count=None,0
    elif candidate==pending:count+=1
    else:pending,since,count=candidate,now,1
    state[k]=(pending,since,count)
    a=audio_index.get(trkey(r))
    if a and a['confirmation_reason']=='duration_confirmed' and r['confirmation_reason']=='hold_or_wait' and candidate is not None and count>=2:
        elapsed=now-since
        if .49<elapsed<.5-1e-10:
            local=dict(fusion=r['fusion'],rule=r['rule'],input_policy=r['input_policy'],episode_id=r['episode_id'],label=episodes[int(r['episode_id'])]['label'],audio_end_s=r['audio_end_s'],elapsed_seconds=elapsed,short_of_confirmation_ms=(.5-elapsed)*1000,display_differs=a['displayed_id']!=r['displayed_id'])
            near.append(local)
    if a and a['displayed_id']!=r['displayed_id']:different.append(r)
if near:save('T2_확인시간_경계사례.csv',near)

wd=read('D_wait_all','trace');bd=read('D_bounded','trace');wi={trkey(r):r for r in wd}
by=defaultdict(list)
for r in bd:
    w=wi[trkey(r)]
    for field in ['displayed_id','raw_top_id','accepted']: assert w[field]==r[field]
    delay=float(w['decision_time_s'])-float(r['decision_time_s'])
    assert delay>=-1e-12
    by[setting(r)].append((delay, int(r['models_ready_at_decision'])))
bounded=[]
for k,values in by.items():
    ds=np.array([v[0] for v in values]);early=ds[ds>1e-12]
    bounded.append(dict(fusion=k[0],rule=k[1],windows=len(ds),certified_windows=sum(v[1]<3 for v in values),
                        mean_saved_ms=float(ds.mean()*1000),median_saved_ms=float(np.median(ds)*1000),
                        p95_saved_ms=float(np.quantile(ds,.95)*1000),mean_saved_when_early_ms=float(early.mean()*1000)))

analysis=dict(verified_settings=sum(map(len,summary.values())),verified_episode_rows=sum(map(len,details.values())),
              fingerprint=fingerprint,adaptive_paired=paired,region_window_counts=dict(region_counts),top_confusions=confusion_rows[:12],
              temporal_boundary=dict(near_confirm_windows=len(near),different_display_windows=len(different),
                                     near_confirm_with_display_difference=sum(r['display_differs'] for r in near),
                                     median_short_of_confirmation_ms=float(np.median([r['short_of_confirmation_ms'] for r in near])),
                                     affected_group_count=len({episodes[int(r['episode_id'])]['group_id'] for r in near}),
                                     example=near[:3]),bounded=bounded)
(OUT/'상세분석.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(analysis,ensure_ascii=False,indent=2))
print('KEY_LABELS',json.dumps([r for r in class_rows if r['stage']=='B_adaptive_fifo' and r['input_policy']=='I3' and r['label'] in ['fire_alarm','knock','baby_cry','dog_bark','background']],ensure_ascii=False))
