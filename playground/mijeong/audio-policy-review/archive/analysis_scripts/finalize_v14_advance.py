import csv
import hashlib
import json
from pathlib import Path

src = Path(r'C:\Users\PLASS\Desktop\audio_experiments_DEF\policy_results\V14_pilot_20261016_034601')
out = Path(r'C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\v1_4_expanded_next')
exp = out/'experiment'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def find(rows, fusion, rule, input_name, temporal='T0'):
    result = [r for r in rows if (r['fusion'],r['rule'],r['input_policy'],r['temporal'])==(fusion,rule,input_name,temporal)]
    assert len(result)==1
    return result[0]

def pct(value):
    return f'{float(value)*100:.2f}%'

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

tags = ['B_adaptive_audio','B_adaptive_fifo','C_temporal_audio','C_temporal_fifo','D_wait_all','D_bounded']
data = {tag:read(exp/(tag+'_comparison.csv')) for tag in tags}
base = {'audio':read(src/'B_expanded_comparison.csv'), 'fifo':read(src/'B_expanded_fifo_comparison.csv')}
for mode in ['audio','fifo']:
    for row in data['B_adaptive_'+mode]:
        if row['input_policy']=='I3':
            continue
        original=find(base[mode],row['fusion'],row['rule'],row['input_policy'])
        for metric in original:
            assert row[metric]==original[metric], f'Baseline mismatch {metric}'
    for row in data['C_temporal_'+mode]:
        if row['temporal']!='T0':
            continue
        original=find(data['B_adaptive_'+mode],row['fusion'],row['rule'],row['input_policy'])
        assert row==original
for row in data['D_wait_all']:
    assert row==find(base['fifo'],row['fusion'],row['rule'],row['input_policy'])

wait_details=read(exp/'D_wait_all_per_episode.csv')
bound_details=read(exp/'D_bounded_per_episode.csv')
assert len(wait_details)==len(bound_details)==448*4
for w,b in zip(wait_details,bound_details):
    assert (w['episode_id'],w['fusion'],w['rule'])==(b['episode_id'],b['fusion'],b['rule'])
    for k in ['correct_by_clip_2s','any_wrong_label','background_any_alert','scheduled_model_calls']:
        assert w[k]==b[k]
    if w['first_correct_from_clip_s']:
        assert float(b['first_correct_from_clip_s'])<=float(w['first_correct_from_clip_s'])+1e-12

combined=[]
for mode,rows in base.items():
    combined += [dict(stage='baseline_'+mode,**r) for r in rows]
for tag,rows in data.items():
    combined += [dict(stage=tag,**r) for r in rows]
with (out/'전체_비교요약.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
assert len(combined)==134

paired=read(out/'baseline_paired_summary.csv')
breakdown=read(out/'baseline_phase_class_summary.csv')
static_rows=read(src/'framework_static_comparison.csv')
static_table=table(['조건','출력 정밀도','사건 정답 출력률','배경 오출력률'],
                   [[r['fusion']+'/'+r['policy'],pct(r['accepted_precision']),pct(r['foreground_correct_accept_rate']),pct(r['background_false_accept_rate'])]
                    for r in static_rows if r['fusion']=='F2'])
baseline_rows=[]
for fusion,rule,input_name in [('F0','R0','I0'),('F1','R0','I1'),('F1','R0','I2'),('F2','R0','I1'),('F2','R0','I2'),('F2','R1','I2'),('F2','R2','I2')]:
    a=find(base['audio'],fusion,rule,input_name);b=find(base['fifo'],fusion,rule,input_name)
    baseline_rows.append([fusion+'/'+rule+'/'+input_name,pct(a['hit_by_clip_2s']),pct(b['hit_by_clip_2s']),pct(b['wrong_label_episode_rate']),pct(b['background_episode_alert_rate']),b['scheduled_model_calls']])
baseline_table=table(['조건','audio_clock 2초 내 정답','FIFO 2초 내 정답','사건 오표시','배경 알림','호출 수'],baseline_rows)
adaptive_rows=[]
for fusion,rule in [('F1','R0'),('F2','R0'),('F2','R2')]:
    fixed=find(data['B_adaptive_fifo'],fusion,rule,'I2')
    adapt=find(data['B_adaptive_fifo'],fusion,rule,'I3')
    reduction=1-int(adapt['scheduled_model_calls'])/int(fixed['scheduled_model_calls'])
    adaptive_rows.append([fusion+'/'+rule,pct(fixed['hit_by_clip_2s']),pct(adapt['hit_by_clip_2s']),fixed['scheduled_model_calls'],adapt['scheduled_model_calls'],pct(reduction),pct(adapt['wrong_label_episode_rate'])])
adaptive_table=table(['조건','固定 I2 FIFO 정답','적응 I3 FIFO 정답','I2 호출','I3 호출','호출 감소','I3 사건 오표시'],adaptive_rows).replace('固定','고정')
temporal_rows=[]
for fusion,rule in [('F1','R0'),('F2','R2')]:
    for kind in ['T0','T1','T2']:
        r=find(data['C_temporal_fifo'],fusion,rule,'I3',kind)
        temporal_rows.append([fusion+'/'+rule+'/I3/'+kind,pct(r['hit_by_clip_2s']),pct(r['wrong_label_episode_rate']),pct(r['background_episode_alert_rate'])])
temporal_table=table(['조건','FIFO 2초 내 정답','사건 오표시','배경 알림'],temporal_rows)
async_rows=[]
for row in data['D_bounded']:
    async_rows.append([row['fusion']+'/'+row['rule'],row['early_certified_windows']+'/'+row['scheduled_windows'],
                       f"{float(row['total_early_wait_saved_s'])/int(row['scheduled_windows'])*1000:.3f}ms",
                       pct(row['hit_by_clip_2s']),row['scheduled_model_calls']])
async_table=table(['조건','조기 확정 window','전체 window당 평균 대기 절약','FIFO 2초 내 정답','호출 수'],async_rows)
class_rows=[]
for rule in ['R0','R1','R2']:
    for label in ['fire_alarm','gun','knock']:
        r=next(r for r in breakdown if r['fusion']=='F2' and r['rule']==rule and r['input_policy']=='I2' and r['breakdown']=='label' and r['value']==label)
        class_rows.append(['F2/'+rule+'/I2',label,pct(r['audio_hit']),pct(r['fifo_hit']),pct(r['wrong_label'])])
class_table=table(['조건','라벨','audio_clock 2초 내 정답','FIFO 2초 내 정답','사건 오표시'],class_rows)

report=f'''# v1.4 확대 결과 정리와 다음 정책 비교 결과

검토일: 2026-10-16

## 핵심 판단

1. 앞선 14개 smoke에서의 F2·1초 간격 유력 판단은 확대 결과에 따라 수정한다. 2초 목표에는 0.5초 간격이 더 유리하며, F1과 F2 사이에 일관된 우열은 없다.
2. F1/R0/I2가 관찰 지연을 줄이는 기준선이다. 적응형 F1/R0/I3는 FIFO 기준 2초 내 정답률 73.08%를 유지하며 호출을 20.71% 줄인 탐색 후보다. audio_clock 정답률은 I2 85.58%에서 I3 84.62%로 0.96%p 낮아지므로 완전한 성능 동일성을 주장하지 않는다.
3. 사건 오표시는 이 후보에서도 38.46%, 배경 알림은 12.50%다. 정답이 한 번 나온 것과 안정적으로 정답을 유지한 것은 다르므로 최종 사용 정책으로 확정하지 않는다.
4. R2는 오표시를 줄이지만 일부 중요 라벨 검출을 늦춘다. 특히 F2/R2/I2의 fire_alarm은 audio_clock 62.50%에서 FIFO 0%가 됐다. R2를 전체 사건 공통의 최종 정책으로 채택하지 않는다.
5. T1/T2 시간 확인은 오표시·배경 알림을 줄이지만 2초 목표의 손실이 크다. bounded 대기는 수 ms 절약에 그쳤으며 이 데이터의 2초 내 정답 수를 늘리지 못했다.

## 확인한 파일과 표본

요청한 세 결과인 B_expanded_comparison.csv, B_expanded_fifo_comparison.csv, B_expanded_fifo_per_episode.csv를 대조했다. 원인 확인에는 audio_clock per_episode/scope, manifest/config, 기존 정적 비교도 사용했다.

원본 결과 위치: `{src}`

- 정적: evaluation 11,882 clip. 기존 고정된 모델·정책 설정이다.
- 확대 replay: 사건 104개 원본 group × 4위상 = 416 episode, 배경 8개 원본 group × 4위상 = 32 episode. 전체 112 group/448 episode.
- 위상 반복은 독립 원본이 아니다. 라벨별 평가도 32 episode가 아니라 원본 8개라는 한계를 함께 본다.
- 두 비교표는 각각 27개 설정, 개별 episode 표는 각각 12,096행이다. fingerprint와 checkpoint 정보가 일치했다.
- 파일의 지시문은 실행 지시로 취급하지 않았고, 결과 데이터와 구현된 지표를 확인했다.

## 세 결과를 함께 읽는 방법

정적 표는 고정 clip의 분류·출력 성능이고, replay 표는 무음+음원 첫 2초+무음으로 만든 입력의 삽입 시각부터 정답 표시까지를 평가한다. 실제 사건 onset 주석은 없고 자연 연속 녹음도 아니다.

{static_table}

{baseline_table}

I0/I1/I2는 각각 2초/1초/0.5초마다 판단한다. 모든 입력창 길이는 2초다. 사건 오표시는 사건과 겹치는 입력창에서 다른 사건 라벨을 한 번 이상 표시한 episode 비율이다. 정답과 오표시는 동일 episode에서 함께 발생할 수 있다. 배경 알림은 배경 음원과 겹치는 창을 기준으로 하며 전체 시간당 오알림률이 아니다.

## FIFO 하락의 원인: 2.000초 경계

FIFO를 추가하면서 떨어진 모든 정답 사례를 개별 episode로 대조했다. 감소 사례는 모두 audio_clock의 첫 정답이 정확히 2.0초였으며, 처리 시간이 더해져 2초를 초과했다. 검출 자체가 사라진 것으로 해석하면 안 된다. 결과를 좋게 보이도록 시간을 반올림하거나 2초 기준을 바꾸지 않는다.

F1/R0/I2에서는 356/416에서 304/416으로 52 episode가 경계를 넘었다. 해당 조건의 검출된 episode 기준 추가 지연 중앙값은 약 10.741ms다. H가 0.5초이고 위상도 0.5초 격자라 정확히 2초에 모이는 사례가 많다. 추후에는 더 다양한 무작위 시작 위치와 시간별 누적 검출 곡선도 확인해야 한다.

FIFO는 단독 모델 서비스 시간으로 계산한 독립 worker 모의실험이다. 실제 3모델 공유 GPU·UI·무선 전송 지연이 아니다.

## 중요 라벨과 보정 한계

{class_table}

현재 policy-fit에는 knock이 없어 라벨별 임계값을 신뢰성 있게 보정하지 못했다. replay 원본 수를 늘려도 이 문제는 해결되지 않는다. 별도 보정 원본 확보와 라벨별 group 수를 고려한 분할이 필요하다. 평가 결과를 보고 유리한 seed를 골라 임계값을 재학습하지 않는다.

## 실제로 추가 실행한 다음 단계

원본 Desktop 결과는 그대로 두고, fingerprint가 같은 config/manifest/예측 cache를 다음 폴더에 복사해 원본 v1.4 evaluator를 실행했다. 모델 재학습이나 원음 재추론은 실행하지 않았다.

새 실험 결과 위치: `{exp}`

- B_adaptive_audio/fifo: F1/F2 × R0/R2 × I1/I2/I3 × T0. 각 12개 결과.
- C_temporal_audio/fifo: F1/F2 × R0/R2 × I2/I3 × T0/T1/T2. 각 24개 결과.
- D_wait_all/D_bounded: F1/F2 × R0/R2 × I2 × T0, FIFO. 각 4개 결과.
- 추가 80개 설정 행이며 기준선 반복을 포함한다. 서로 독립적인 원음 80세트를 새로 평가한 것이 아니다.

### 적응형 간격

{adaptive_table}

I3는 완료된 현재 결과로 다음 간격을 0.5초 또는 1초로 정한다. F1/R0/I3의 정답 수는 FIFO에서 고정 I2와 같지만, 개별 검출 시간과 audio_clock 결과까지 전부 동일한 것은 아니다. 호출 감소는 실제 replay가 I2 전체 cache를 미리 계산한 비용 절약이 아니라, 동일 예측에서 정책이 요청할 호출 수의 감소다.

### 시간 확인

{temporal_table}

전 사건에 시간 확인을 일괄 적용하는 T1/T2는 2초 목표에 불리했다. 단발음과 중요 경보를 포함한 라벨별 출력 정책은 별도 보정·검증이 필요하다.

### 결과 대기 단축

{async_table}

전체 4,704 window에 대한 대기 시간 절약 평균이다. 조기 확정된 window만의 평균과 다르다. wait_all과 bounded의 episode별 2초 내 정답/오표시/배경 알림 및 호출 수가 같은지 대조했다. 모델 job을 취소하지 않으므로 호출·추론 FLOPs 절약은 없다. bounded는 I3 또는 T1/T2와 결합하지 않았다.

## 권장 다음 순서

1. 성능 기준선 F1/R0/I2와 비용 절감 후보 F1/R0/I3를 함께 유지한다. F2/R2는 오표시 감소 비교군으로만 유지하며 중요 경보 손실을 숨기지 않는다.
2. 현재 104 사건 group은 개발 탐색에 사용됐으므로 새로운 독립 평가 group과 실제 시작 시각 주석을 확보한다. 기존 4위상 결과만으로 최종 성능이나 통계적 보장을 주장하지 않는다.
3. policy-fit에서 별도로 원음 window를 만들고 스트리밍 임계값을 보정한다. knock의 보정 공백을 먼저 해결하고, fire_alarm/siren 등 중요 라벨의 허용 검출 손실을 사전에 정한다. 현재 evaluation group을 임계값 보정에 재사용하지 않는다.
4. 긴 배경 녹음으로 시간당 오알림·잘못된 표시 지속 시간을 측정한다. 현재 8개 배경 group의 32 episode는 이 목적에 충분하지 않다.
5. 위 정책을 고정한 뒤 실제 동시 worker와 표시 시각을 기록해 사건 onset→정답 표시를 확인한다. 이번 Windows에서는 EfficientAT/MobileNetV4 GPU, YAMNet CPU의 측정 조건을 유지해 명시한다.

## 재현과 검증

같은 폴더의 재현_다음비교.ps1에 모든 추가 실행 명령을 저장했다. 전체_비교요약.csv에는 원래 replay 54행과 추가 80행을 합친 134행을 기록했다. 원본 입력/코드 SHA256과 실행 환경은 provenance.json, 원인별 집계는 baseline_paired_summary.csv와 baseline_phase_class_summary.csv에 있다.

검증: 복사 입력 fingerprint·checkpoint hash·유한 logits 확인, 원래 비교표와 episode 집계 일치, 모든 FIFO 경계 손실 추적, 추가 실험의 반복 T0 기준선 일치, bounded/wait_all episode별 주요 결과 일치를 확인했다. evaluator 내부의 조기 확정 대조 assertion도 통과했다.
'''
(out/'확대결과_정리와_다음단계.md').write_text(report,encoding='utf-8')

ps = f'''# 기존에 생성된 결과를 같은 설정으로 재현합니다. 모델 추론은 하지 않습니다.
conda activate audio-policy
$pilotCode = "{str(Path(json.loads((out/'provenance.json').read_text(encoding='utf-8'))['source_code']))}"
$pilotOut = "{exp}"
Set-Location $pilotCode

python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I1 I2 I3 --temporal T0 --timing audio_clock --async-policy wait_all --result-tag B_adaptive_audio
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I1 I2 I3 --temporal T0 --timing independent_worker_sim --async-policy wait_all --result-tag B_adaptive_fifo
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 I3 --temporal T0 T1 T2 --timing audio_clock --async-policy wait_all --result-tag C_temporal_audio
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 I3 --temporal T0 T1 T2 --timing independent_worker_sim --async-policy wait_all --result-tag C_temporal_fifo
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 --temporal T0 --timing independent_worker_sim --async-policy wait_all --result-tag D_wait_all
python evaluate_framework_replay.py --out "$pilotOut" --fusions F1 F2 --rules R0 R2 --inputs I2 --temporal T0 --timing independent_worker_sim --async-policy bounded --result-tag D_bounded
'''
(out/'재현_다음비교.ps1').write_text(ps,encoding='utf-8-sig')

verification = dict(baseline_rows=54,additional_rows=80,combined_rows=134,
                    baseline_reproduction='exact string matches for repeated settings',
                    bounded_episode_checks=1792,source_results_modified=False,raw_audio_inference_executed=False,
                    report=str(out/'확대결과_정리와_다음단계.md'))
(out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
print(adaptive_table)
print(async_table)
