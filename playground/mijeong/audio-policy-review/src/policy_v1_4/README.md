# 입력·출력 정책 실험 코드 v1.4

> **v1.4 통합 정책 연구:** [정책_재설계_및_통합실험계획_v1_4.md](정책_재설계_및_통합실험계획_v1_4.md)가 최신 계획·근거·실행 안내입니다. 무투표 F0를 기준으로 평균 F1·클래스 가중 F2, 입력 I, 확정 R, 시간 확인 T, 비동기 확정 A를 분리해 비교합니다. 새 시작점은 `run_framework_pilot.py`, 회귀 평가는 `evaluate_framework_replay.py`입니다. 원음 GPU export 경로는 v1.3의 `--selection`을 재사용합니다. 이전 결과 폴더를 덮어쓰지 말고 새 폴더에서 시작하세요.

> **v1.3 입력 실험 연결:** [INPUT_POLICY_EXPERIMENT.md](INPUT_POLICY_EXPERIMENT.md)가 최신 실행 안내입니다. `prepare_replay.py --selection` → `check_replay_audio.py` → `export_replay_predictions.py --selection` → `evaluate_selection_replay.py`를 사용합니다. 문맥 2초를 유지한 H2/1/0.5초 비교를 무투표 S0에 연결했습니다. 신규 원음·GPU 성능 검증은 실행 환경에서 진행해야 합니다. 아래 기존 `evaluate_replay.py`는 가중 voting 참고 경로입니다.

> **v1.2 추가:** 무투표 담당 라벨 최댓값 선택은 **run_selection_pilot.py**와 [SELECTION_POLICY_PLAN.md](SELECTION_POLICY_PLAN.md)를 사용합니다. S0 원점수 max / S1 공통 기준 / S2 라벨별 기준을 비교하며 모델 간 평균·가중합·재정규화를 하지 않습니다. 아래 v1.1 실행 순서는 가중 voting 참고 실험용으로 남겨 둡니다. background의 실제 서비스 처리 경로는 제공 소스에서 확인되지 않아 새 실행에서는 명시적인 민감도 옵션으로 분리합니다.

> 기준 구조 정정: 사용자가 확인한 기존 최고 성능 방식은 모델별로 전용 라벨을 다르게 학습한 **voting 없는 specialist 구성**이다. 이 v1.1의 가중 fusion은 그 방식의 재현이 아니라 별도 비교 구현이다. 아래 실행 결과·임계값을 실제 최고 성능 방식의 검증 결과로 해석하거나 그대로 적용하지 않는다. 제공된 오프라인 소스에서는 해당 무투표 구성의 최종 단일 라벨 선택 경로를 확인하지 못했으므로, 이를 확인한 뒤 기존 선택 규칙을 보존한 정책 비교를 구성해야 한다.

모델 내부 early exit 없이 기존 14-class specialist 가중 fusion을 사용합니다.
원본 프로젝트를 덮어쓰지 않는 별도 실행 코드입니다.

## 1. 비교군

### 입력 정책: 문맥 2초 고정

| 이름 | 실행 간격 | 입력 겹침 |
|---|---:|---:|
| I0 | 2초 | 0% |
| I1 | 1초 | 50% |
| I2 | 0.5초 | 75% |

### 출력 정책

| 이름 | 규칙 |
|---|---|
| O0 | 기존 전체 14-class argmax. background면 알림 없음 |
| O1 | O0의 foreground 최고 후보가 공통 임계값 0.6 이상일 때만 출력 |
| O2 | O0의 foreground 최고 후보가 별도로 설정한 라벨별 임계값 이상일 때만 출력 |

O2는 validation의 그룹별 분할 중 설정용 부분만 사용하여,
정밀도 목표 95%·출력 수 20개 이상 조건에서 올바른 출력 수를 최대화합니다.
해당 조건을 만족하는 임계값이 없으면 그 라벨은 이 실험에서 미확정 처리합니다.
policy_config.json의 null이 그 뜻입니다. 높은 목표를 만족하지 못하는 라벨을
강제로 출력하지 않는 비교군이며, 위험음 운영 정책으로 바로 채택하는 설정이 아닙니다.
O1의 0.6과 O2의 95%는 초기 비교용 값입니다. CLI로 변경할 수 있습니다.

단일 출력은 전체 최고 라벨을 먼저 정한 뒤 확정 여부를 검사합니다.
낮은 순위 후보로 대체하지 않습니다. 이번 버전은 연속 확인·해제·학습형 정책을 추가하지 않습니다.
우선 cutoff 차이와 실행 간격의 효과만 확인합니다.

## 2. 필요한 기존 파일과 환경

기존 프로젝트는 하나로 합쳐진 정상 실행 폴더를 사용하세요.
예: /home/plass/YAMNet/audio_experiments_DEF

- class_names.txt, metadata.csv
- common_audio.py, common_torch.py, common_metadata.py, evaluation_utils.py, config.py
- train_efficientat.py, train_mobilenetv4.py
- third_party/EfficientAT 전체
- runs/E_predefined_specialists/nonexpert_070/{각 모델}/best.pt 또는 best.keras
- 위 모델들의 val_predictions.npz
- runs/F_predefined_weighted_ensemble/nonexpert_070/classwise_model_weights.csv
- metadata.csv에서 참조하는 실제 음원

새 코드 자체의 추가 라이브러리는 NumPy입니다. 기존 학습 환경에 이미 있는
Torch/timm/torchaudio, TensorFlow/tensorflow_hub, scipy/soundfile 등을 재사용합니다.
프레임워크를 한 환경에 새로 통합하지 마세요. 기존 requirements의 버전을 유지하세요.

본 코드의 기본 대상은 predefined/0.70입니다.
automatic을 쓰려면 static 단계에서 --assignment automatic을 사용하세요.
후속 replay는 저장된 policy_config.json의 설정을 자동으로 따릅니다.

## 3. 실행 순서: 기존 Linux GPU 환경

압축 해제한 코드 폴더에서 실행합니다.
PILOT_PROJECT만 실제 기존 프로젝트 위치에 맞추세요.
PILOT_OUT은 실험 결과가 생성될 새 폴더입니다.

### A. 정적 출력 정책 비교와 원음 목록 생성

기존 Torch 환경에서:

~~~bash
conda activate audio-torch
cd /압축해제한경로/policy-pilot

PILOT_PROJECT=/home/plass/YAMNet/audio_experiments_DEF
PILOT_OUT="$PILOT_PROJECT/pilot_results_v1"

python -m unittest discover -v

python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --out "$PILOT_OUT" \
  --verify-saved-test-fusion

python prepare_replay.py \
  --out "$PILOT_OUT" \
  --max-per-class 8
~~~

--verify-saved-test-fusion은 기존 test fusion과의 수치 호환만 확인합니다.
test로 임계값을 조정하지 않습니다. 저장된 test_predictions.npz가 없다면 옵션을 빼세요.

정적 비교는 GPU 없이도 실행됩니다.
metadata의 원음 경로가 현재 환경에서 그대로 유효하면 dataset-root 옵션이 필요 없습니다.
데이터가 이동했다면:

~~~bash
python prepare_replay.py \
  --out "$PILOT_OUT" \
  --max-per-class 8 \
  --old-dataset-root /home/plass/YAMNet/aug_data \
  --dataset-root /현재/음원/루트
~~~

처음 연결 점검은 --max-per-class 1로 실행하고,
정상 동작하면 8 이상으로 늘려 전체 모델 export를 다시 실행해도 됩니다.
manifest를 바꾸면 세 모델의 replay 결과를 모두 다시 생성하세요.

### B. Torch 모델 두 개의 GPU replay

같은 Torch 환경에서:

~~~bash
python export_replay_predictions.py \
  --project-root "$PILOT_PROJECT" \
  --out "$PILOT_OUT" \
  --model efficientat \
  --device cuda

python export_replay_predictions.py \
  --project-root "$PILOT_PROJECT" \
  --out "$PILOT_OUT" \
  --model mobilenetv4 \
  --device cuda
~~~

EfficientAT 기본 모델은 현재 소스의 mn04_as입니다.
다른 width로 학습했다면 --efficientat-model-name을 실제 checkpoint와 맞추세요.
checkpoint의 클래스 순서와 weight shape는 검사합니다.

### C. YAMNet GPU replay

기존 YAMNet 환경으로 바꿉니다. 같은 코드 폴더와 PILOT_OUT을 사용하세요.

~~~bash
conda activate audio-yamnet
export TF_ENABLE_ONEDNN_OPTS=0

python export_replay_predictions.py \
  --project-root "$PILOT_PROJECT" \
  --out "$PILOT_OUT" \
  --model yamnet \
  --device cuda
~~~

TensorFlow가 GPU를 찾지 못하면 오류로 중단합니다.
CPU 결과가 필요한 경우에만 --device cpu로 바꾸세요.
YAMNet은 기존처럼 tfhub 모델을 로드하므로 기존 캐시 또는 네트워크 접근이 필요합니다.
GPU 메모리 경쟁을 피하도록 세 export를 순차 실행하는 것을 기준으로 합니다.

### D. 입력 3 × 출력 3의 결과 비교

NumPy가 있는 어느 환경에서든:

~~~bash
python evaluate_replay.py --out "$PILOT_OUT"
~~~

같은 0.5초 기본 격자의 window 결과에서 1초·2초 실행 위치를 추출합니다.
그래서 모델 export는 세 번만 필요하며 9가지 조합별로 모델을 다시 돌릴 필요는 없습니다.
성능 비교의 입력 window는 동일합니다. 실제 운용 주기에 따른 발열·큐·동시성 차이는 별도 실험입니다.

## 4. 원음을 어떻게 재생하는가

기존 평가 그룹에서 각 라벨당 최대 8개 원본 group을 선택합니다.
같은 group의 증강 clip 여러 개를 독립 원음처럼 뽑지 않습니다.

한 episode는 다음과 같습니다.

~~~text
2초 zero silence + 기존 음원의 선두 2초 + 2초 zero silence
~~~

현재 전처리를 window마다 그대로 재현합니다.
80Hz high-pass, 100ms max-RMS, -30dBFS 기준 정규화 여부, -20dBFS 목표값입니다.
무음이라도 추론을 생략하지 않습니다.

파일 읽기·원본 sample-rate 변환은 episode 준비 단계이며 window별 서비스 시간에 포함하지 않습니다.
replay는 변환된 16kHz 오디오를 기준 입력 신호로 취급합니다.
window별 시간에는 high-pass·RMS·정규화 및 모델 frontend/embedding/head를 포함합니다.

현재 checkpoint 학습 시 전처리 인자를 바꿨다면 replay_manifest.json의 audio_config를
그 학습 설정에 맞춘 뒤 세 export를 다시 실행하세요.
이 실험에서 threshold·hop과 전처리 변경을 동시에 섞지 않는 것이 좋습니다.

zero silence는 통제된 실험 입력입니다. 실제 실내 background를 대표하지 않습니다.
clip의 실제 사건 시작 시각은 제공되지 않았으므로
결과의 기준은 'clip 삽입 시점'이며 '실제 사건 onset'이 아닙니다.
대상 음원이 시작된다는 사실만으로 그 순간부터 사건이 난다고 가정하지 않습니다.

## 5. 결과 파일 읽기

| 파일 | 내용 |
|---|---|
| policy_config.json | 그룹 분할 수, 고정 가중치, O1/O2 설정 |
| static_policy_comparison.csv | 저장된 정적 validation의 출력 정책 3개 비교 |
| static_per_class.csv | 라벨별 정답 출력·거부 성능 |
| threshold_fit.csv | 설정 데이터에서 찾은 임계값 |
| compatibility_checks.json | 기존 fusion 일치와 그룹 중복 검사 |
| static_scores.npz | 재현용 점수·그룹·설정/평가 mask |
| replay_manifest.json | 원음 목록·입력·전처리 조건 |
| replay_각모델.npz | 새 원음 replay의 logits와 소요시간 |
| runtime_각모델.json | GPU·프레임워크·batch=1 시간 |
| replay_policy_comparison.csv | 입력 3 × 출력 3 비교 |
| replay_per_episode.csv | clip별 첫 정답·오출력 기록 |
| replay_scope.json | 해석 범위 |

정적 지표:
- accepted_precision: 출력한 foreground clip 중 정답 비율.
- foreground_correct_accept_rate: 실제 foreground 전체 중 올바르게 출력한 비율.
- background_false_accept_rate: background clip을 foreground로 출력한 비율.

replay 지표:
- foreground_hit_by_insert_2s_audio_time:
  clip 삽입 후 2초 이내의 오디오 시각에서 올바른 출력을 얻은 episode 비율.
- first_hit_audio_time_s_*:
  검출된 episode만의 삽입 기준 첫 정답 시각. 미검출 수와 함께 확인합니다.
- background_episode_any_alert_rate:
  background 원음 episode에서 잘못된 foreground 출력이 있었던 비율.
- cached_serial_service_s:
  선택한 실행 window들의 모델별 단독 처리시간 합.
- cached_step_serial_service_ms_p95:
  같은 window의 세 단독 처리시간을 더한 값의 p95.

시간 지표는 실제 UI 표시 지연이 아닙니다.
동시 실행·IPC·큐·디스플레이 지연과 실제 event onset 지연을 주장하지 않습니다.
같은 배포 GPU에서 실행하더라도 측정된 것은 개별 worker의 batch=1 서비스 시간입니다.

## 6. 확인한 초기 결과

제공된 F/nonexpert_070 validation을 원본 group hash(seed42)로 나눴습니다.

- 설정: 11,965 clip / 1,351 group
- 평가: 11,882 clip / 1,338 group
- group 교차: 0

| 출력 정책 | 출력 정답률 | foreground 정답 출력률 | background 잘못된 출력률 |
|---|---:|---:|---:|
| O0 | 83.66% | 84.77% | 17.37% |
| O1 | 96.46% | 66.92% | 5.09% |
| O2 | 95.31% | 69.10% | 5.36% |

원 모델 선택과 fusion weight가 이미 기존 validation을 사용했습니다.
이번 group 분할은 출력 threshold의 설정/평가를 분리한 탐색이며
전체 시스템의 완전히 독립된 최종 검증이나 통계적 보장이 아닙니다.
예를 들어 현재 분할에서 O2의 fire_alarm/knock 임계값이 null일 수 있습니다.
데이터 부족·목표 조건의 엄격함에 따른 거부이며 운영 결론이 아닙니다.
원 데이터 비율·대상 실험을 바꾸면 숫자도 바뀝니다.

확인한 검증:
- 핵심/평가/배정 확장 unittest 14개 통과.
- 기존 F70 test fusion top-1 일치율 100%.
- 재계산 확률의 최대 절댓값 차이 약 2.22e-16.
- 모든 Python 파일 문법 검사.
- 실제 GPU model export는 이 코드 작성 환경에서 실행하지 못했습니다.

example_results는 확인한 작은 결과 파일만 포함합니다.
사용자 GPU 환경에서 run_static_pilot.py를 실행하면 replay 준비에 필요한
static_scores.npz가 새로 생성됩니다.

## 7. 실시간 코드에 연결하는 위치

policy_core.py의 SlidingWindowScheduler와 WindowFusionCoordinator를 사용합니다.

~~~python
scheduler = SlidingWindowScheduler(
    sample_rate=16000, window_seconds=2.0, hop_seconds=0.5
)

# 실제 수신 loop:
for window in scheduler.push(new_audio_chunk):
    # 원래 전처리·모델을 유지한 세 상주 worker에 같은 window를 전달
    dispatch_to_existing_workers(window)

# 단일 결과 소비 loop에서:
result = coordinator.submit(
    window_id, model_name, logits, start_sample, end_sample
)
if result is not None and result["accepted"]:
    display_one_label(class_names[result["top_id"]])
~~~

dispatch/display는 사용자 기존 runtime의 연결 지점입니다.
본 묶음은 마이크·Bluetooth·UI 서버 자체를 구현하는 패키지가 아닙니다.
coordinator는 단일 소비 loop용이며 내부 lock이 없는 API입니다.
이번 pilot은 같은 window의 세 모델을 기다려 기존 fusion을 보존합니다.
동시 실행 모델의 순위 bound나 연속 확인·해제 정책은 다음 실험 단계로 남깁니다.

## 8. 재현 설정을 바꾸는 예

~~~bash
python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --out "$PILOT_PROJECT/pilot_results_threshold065" \
  --global-threshold 0.65 \
  --target-precision 0.95 \
  --minimum-accepted 20 \
  --seed 42
~~~

파라미터를 바꾸면 별도 output 폴더를 사용하세요.
실험 후 static_policy_comparison.csv, static_per_class.csv,
replay_policy_comparison.csv, runtime_*.json을 함께 비교하면 됩니다.



## 9. v1.1: 정책 우선, 전문 클래스 재분류는 후속 단계

[LABEL_REGROUPING_PLAN.md](LABEL_REGROUPING_PLAN.md)에 사전 기준·후보 생성·정식 비교 계획을 정리했습니다.

- 현재: R0 기존 배정에서 I0/I1/I2와 O0/O1/O2 실험.
- 추후: R0 기존 수동 / R1 기존 F1 자동 / R2 음향·혼동·관찰 기반 후보 비교.
- 재분류 후: 기존 정책값 그대로 이전한 경우와 같은 정책의 cutoff를 재설정한 경우를 분리.
- 라벨 이름·14-class 출력은 유지. expert 배정은 별도 데이터 설정으로 관리.

추가 파일:
- label_profiles.csv: 13개 라벨의 초기 가설과 미측정 항목.
- experiments/R0_predefined_070.json: 기존 배정의 설정 예.
- experiments/R1_f1_automatic_070.json: 저장된 자동 배정의 설정 예.
- experiments/R2_custom_template.json: 아직 선정·학습되지 않은 후속 실험 템플릿.
  ready_for_evaluation=false로 잠겨 있으며 실제 평가에 사용되지 않습니다.

추가 옵션:
- --experiment-spec: 미래에 새로 학습한 specialist와 weight 경로 및 배정표 지정.
- --policy-config-from: 이전 실험의 출력 cutoff만 가져옴.
  새 모델의 weight를 사용하며, 이전 모델 weight를 복사하지 않습니다.

라벨 identity/순서가 바뀌면 정책 이전을 중단합니다.
새 config는 experiment_spec, weight hash, 평가 sample digest, cutoff 출처를 기록합니다.
GPU replay worker도 해당 config의 실험 경로를 사용합니다.

이미 v1로 GPU 실험을 진행 중이면 같은 config와 코드로 해당 실행을 마무리하고 결과를 보존하세요.
새 버전은 별도 output 폴더에서 실행하는 편이 좋습니다.
모델·배정 변경은 현재 수행하지 않았습니다.

