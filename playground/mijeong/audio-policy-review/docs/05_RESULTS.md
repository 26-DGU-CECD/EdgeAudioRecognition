# 현재 실험 결과와 다음 과제

## 실험 이력

| 단계 | 목적 | 공유 위치 |
|---|---|---|
| S01/S02 | 저장된 점수 기반 owner 선택·임계값과 seed 비교 | results/original/S01_*, S02_* |
| 별도 v1.3 | 원음 연결·selection replay 구현 | archive/policy_v1_3, results/codex/S03_smoke |
| v1.4 smoke | 14원본으로 경로·모델·평가 연결 확인 | results/original/V14_smoke_* |
| v1.4 expanded | 112원본×4위상, F/R/고정 간격 비교 | results/original/V14_pilot_* |
| v1.4 followup | 적응형 I3, 시간 확인 T1/T2, bounded | results/followup |

v1.4는 W=2초를 유지하고 F 결합, R 확정, I 실행 간격, T 시간 확인, A 대기를 비교한다. 새 모델 학습·RL·정책 신경망·모델 내부 early exit 실험은 현재 단계에 없다.

## 주요 결과

- F1/R0/I2/T0: FIFO 모의실험의 삽입 후 2초 내 정답 73.08%, 요청 모델 호출 14,112회.
- F1/R0/I3/T0: 같은 2초 충족 대상을 유지하며 11,190회, 20.71% 호출 감소. 사건 오표시 38.46%, 배경 알림 12.50%가 남는다.
- 처리 시간 제외 기준에서는 scream 원본 1개의 네 위상에서 추가 지연이 있어 정답률이 85.58%→84.62%로 감소한다. 모든 성능이 동일한 것은 아니다.
- 절감된 창의 약 절반은 합성 무음 구간이다. 실제 자연 배경에서 같은 호출·전력 절감이 나오는지 별도 확인해야 한다.
- R2는 오표시를 줄이지만 중요 경보의 2초 검출을 늦춘다. F1/R2/I3의 fire_alarm은 FIFO 2초 내 0/32이며, 그중 20개는 늦게 검출했다.
- T1/T2는 오표시와 배경 알림을 줄이나 2초 목표 손실이 크다. T2는 완료 시각 차이가 0.5초 확인 경계에 걸리는 민감도가 있다.
- bounded는 고정 I2/T0에서 전체 창당 평균 약 1.6~2.2ms 절약. 호출·추론량과 2초 내 정답 수는 같았다. I3나 T1/T2와 결합한 결과는 아니다.

## 다음 단계

1. F1/R0/I2를 성능 기준선, F1/R0/I3를 비용 절감 후보로 유지한다.
2. reports/v14_detailed의 오표시 WAV 목록으로 라벨·실제 시작 시각을 확인한다. 주요 혼동은 car_horn→gun, fire_alarm→siren, dog_bark→cry다.
3. 별도 policy-fit 원본에서 rolling-window 임계값을 보정하고 knock의 보정 공백을 해결한다. evaluation 원본을 보정에 재사용하지 않는다.
4. 정책 고정 후 독립 자연 음원, 긴 배경 녹음, 실제 동시 worker/UI 시각으로 검증한다.

자세한 검산과 원인 분석은 reports/v14_detailed/추가실험_상세분석.md를 본다. 기존 기록의 절대 경로는 당시 PC 기준이다. 현재 공유 폴더의 대응은 docs/01_PATHS_AND_MODELS.md와 manifests/path_map.json에 있다.
