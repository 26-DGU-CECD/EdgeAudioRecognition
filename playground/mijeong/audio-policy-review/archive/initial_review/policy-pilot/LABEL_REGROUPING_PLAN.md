# 정책 우선 구현과 후속 전문 클래스 재분류 실험 계획

> 최신 무투표 선택 정책·background 코드 감사·후속 선택 비교군은 **SELECTION_POLICY_PLAN.md**가 우선합니다. v1.2는 모델별 head 라벨 순서와 expert_classes를 분리하므로, 이후 담당 라벨 재배정 시 전용 head 또는 전체 14-class head 양쪽을 지원합니다. 기존 R0/R1 비교용 가중 fusion 설명을 무투표 선택의 필수 조건으로 해석하지 않습니다.

> 기준 구조 정정: 실제 정책 연구의 기준은 사용자가 확인한 **모델별 전용 라벨 학습·voting 없는 최고 성능 구성**으로 한다. 아래의 14-logit 가중 fusion, F/G 경로, R0_predefined/nonexpert_070 설명은 제공 소스 기반의 가중 voting 비교 구현에만 해당하며 실제 최고 성능 구성의 확정 사양이 아니다. 무투표 구성의 출력 head·단일 라벨 선택 규칙을 확인한 뒤 그 학습·선택 구조를 유지하고 입력·출력 정책만 비교한다. 후속 라벨 재배정에서도 voting을 기본값으로 추가하지 않는다. 현재 v1.1의 결과는 가중 voting 참고 결과로만 보관한다.

## 1. 연구 순서

**1단계: 기존 R0 배정으로 정책 구현·pilot 비교 → 2단계: 전문 클래스 후보 생성·재학습 → 3단계: 정책 이전·재보정 비교**

현재 단계에서 라벨 재분류 학습은 수행하지 않는다. 정책 코드는 각 라벨 이름을 기준으로 동작하고, 전문 배정·모델 결과 경로·fusion weight를 외부 설정으로 받아 이후 변경을 수용한다.

여기서 재분류는 **13개 사건 라벨의 이름·정의는 유지하고 specialist의 expert 역할을 다시 배정하는 것**이다. background 포함 14 logits 출력과 모델 수 3개를 유지한다. 클래스 합치기·삭제·새 클래스 추가는 별도 실험이다.

현재 소스의 전문화는 expert 100%/일부 nonexpert/hard-negative의 학습 표본 유지와 expert fusion boost로 구현된다. backbone이 고정된 현재 학습 조건에서는 주로 head와 샘플링 효과를 검증하는 실험이다. 재배정만으로 추론 FLOPs나 관찰 시간이 줄어든다고 가정하지 않는다.

## 2. 지금의 정책 비교군

| 축 | 비교군 | 고정 조건 |
|---|---|---|
| 입력 | I0=H2초 / I1=H1초 / I2=H0.5초 | W2초, 전처리, 모델 checkpoint |
| 출력 | O0=기존 argmax / O1=공통0.6 / O2=라벨별 cutoff | background 처리·fusion 알고리즘 |

현재 단계는 R0_predefined/nonexpert_070을 기준으로 한다.
기존 자동 배정 R1 결과도 있지만 정책 pilot에 한꺼번에 섞어 winner를 정하지 않는다.

policy_config.json에 저장한 O1/O2는 전문 소속 이름 대신 라벨 이름으로 저장된다.
I0/I1/I2도 라벨별 expert 소속을 하드코딩하지 않는다.
현재 custom spec은 모델 수·정체성(EfficientAT/YAMNet/MobileNetV4)을 유지한 배정 변경을 지원한다.

## 3. 사전 정의할 기준

라벨 프로파일은 “근거가 확인된 측정값”과 “초기 가설”을 분리한다.
label_profiles.csv에는 현재 라벨의 초기 가설과 미측정 필드를 기록했다.
이 파일을 이미 검증된 정책 수치나 자동 분할 결과로 사용하지 않는다.

| 기준 | 측정·확인 방법 | 재분류에서의 역할 |
|---|---|---|
| 라벨 정의 | bicycle의 벨/주행음, cry와 baby_cry 관계 확인 | 잘못 정의된 혼동을 먼저 제거 |
| 시간 형태 | 사건 onset/offset, 단발·반복·지속 분포 | 관찰 특성이 유사한 후보 생성 |
| 음향 형태 | 공통 특징 공간의 음색·조화성·잡음성·변조 | 음향 유사도 후보 생성 |
| 실제 혼동 | 전체 14-class baseline/fusion의 validation 혼동 | 서로 구별할 필요가 큰 라벨 쌍 |
| 모델 적합성 | baseline별 라벨 F1·recall·support | 어느 backbone에 배정할지 |
| 짧은 관찰 적합성 | 동일 오경보 제약의 문맥/진행 구간별 성능 | 관찰 정책과의 결합 가능성 |
| 학습 적합성 | 같은 학습 예산으로 후보 재학습 | 묶었을 때의 이득·부작용 |
| 긴급도 | 기존 주의/긴급/일상 정보 유지 | recall·지연 제약과 평가; 유사도 자체와 구분 |
| 데이터 편중 | 원본 group 수·희소 라벨 recall | 작은 라벨의 성능 저하 방지 |
| 장치 비용 | 같은 하드웨어의 batch=1 처리·에너지 | 향후 최종 배치 선정 |

2초로 잘린 파일 길이는 사건 지속시간이 아니다.
현재 metadata에 실제 onset/offset이 없어 지속시간 분포와 최소 필요 관찰 시간은 미측정으로 둔다.
증강 clip 수 대신 원본 group 수를 함께 기록한다.

## 4. 최종 비교군을 정하기 전의 개발 실험

### 4.1 기존 자료로 할 일

- 기존 R0와 자동 배정 R1의 validation 지표를 정리한다.
- 전체 14클래스의 혼동을 분석하여 다른 specialist 라벨 간 혼동도 포함한다.
- 음향 유사도는 공통 encoder로 추출하거나 각 특징 공간의 관계를 따로 비교한다.
  서로 다른 embedding 벡터를 같은 좌표처럼 직접 비교하지 않는다.
- 초기 발성/경보/충격 등은 라벨의 가설 태그로 사용한다.
  이 태그 수를 specialist 수로 그대로 바꾸지 않는다.

### 4.2 R2 후보 생성

R2는 아래 두 초기 후보를 개발 데이터에서 비교한 뒤 **하나를 사전에 선정**한다.

- R2a: 혼동 기반 후보. 자주 오분류되는 클래스를 함께 집중 학습.
- R2b: 음향·관찰 특성 결합 후보. 공통 특징 유사도와 관찰 요구의 관계를 함께 고려.

먼저 3개 specialist, 라벨 단일 expert 소속, 동일 학습 예산을 유지한다.
군집화 결과를 그대로 확정하지 않고, 라벨 이동·교환으로 모델별 난이도·데이터 편중을 조정한다.

후보의 공동 학습 성능, 주요 혼동 쌍, 희소·긴급 라벨 recall을 비교한다.
개발 단계에서 유망한 후보만 충분히 재학습하고, 선택 기준을 기록한 뒤 최종 test를 사용한다.
개발 결과가 R1/R0보다 나쁘면 R2를 개선된 방법으로 취급하지 않는다.

“유사한 라벨은 항상 묶는 것이 좋다”는 전제를 두지 않는다.
현재 backbone 고정 조건에서의 효과를 먼저 확인하고, backbone fine-tuning은 독립 확장으로 둔다.

### 4.3 정식 재분류 비교군: 세 개

| ID | 배정 방법 | 상태 |
|---|---|---|
| R0 | 기존 사전 지정 주의/긴급/일상 배정 | 코드·저장 결과 존재 |
| R1 | baseline validation per-class F1 최고 모델 배정 + 최소 담당 수 보정 | 코드·저장 결과 존재 |
| R2 | 개발 실험으로 고른 음향/혼동/관찰 기반 배정 | 향후 설계·재학습 예정 |

R1은 음향 특성을 분석한 재분류로 부르지 않는다.
현재 build_auto_specialist_metadata.py는 F1 기반 배정이며 혼동은 hard-negative 선정에 사용된다.

처음에는 모델–그룹 배정 탐색을 제한하여 예산을 맞춘다.
예산이 허용되면 유망한 3그룹과 3모델의 최대 6개 배정을 비교하되,
한 방법만 더 많은 학습·튜닝 기회를 받지 않도록 기록한다.

## 5. 재분류와 정책 효과를 분리하는 비교

정책 pilot에서 유망한 정책 계열 P*를 고른 뒤 다음을 비교한다.
pilot의 삽입 시각 지표만으로 실환경 2초 목표를 달성했다고 선언하지 않는다.

| 비교 | 배정 | 정책 | 질문 |
|---|---|---|---|
| T0 | R0 | 기존 I0/O0 | 기준 |
| T1 | R0 | P* | 정책 단독 효과 |
| T2 | R1 또는 R2 | 기존 I0/O0 | 재분류 단독 효과 |
| T3 | R1 또는 R2 | R0의 P* 기준값 그대로 | 고정 정책의 이전 가능성 |
| T4 | R1 또는 R2 | 같은 P* 계열, 기준값 재설정 | 재분류 후 보정의 추가 효과 |

T3/T4에서는 입력 H와 출력 규칙의 구조를 유지하고 cutoff만 바꾸는 비교를 우선한다.
입력 주기 자체를 다시 탐색하는 것은 다음의 별도 비교다.

출력 cutoff를 이전할 때 R0의 fusion weight까지 복사하지 않는다.
각 배정에서 같은 F1 weight 산식/floor/power/boost를 사용해 그 모델의 validation으로 만든 weight를 사용한다.
숫자 weight와 threshold의 출처를 각각 기록한다.

기본 재분류 비교는 배정·학습 결과·동일 산식으로 생성한 fusion을 포함한다.
이후 순수한 샘플링 효과를 분리할 필요가 있으면 공통 weight 대조군을 추가한다.

선정 기준은 다음 순서로 사용한다.

1. 라벨별 목표 recall·오경보 제약을 충족하는지 확인.
2. 부족한 라벨을 평균 macro-F1에 가리지 않고 보고.
3. 제약 충족 후보 사이의 지연·처리 비용 비교.
4. 정확도–지연의 Pareto 관계를 보고.
5. 개선이 없으면 해당 후보를 채택하지 않음.

현재 95% precision/최소20출력은 O2 pilot의 임시 설정이다.
제품의 최종 목표나 R2 후보 채택 기준으로 자동 승격하지 않는다.

## 6. 데이터와 통제 조건

- 원본 group 단위 동일 split·동일 평가 sample을 사용한다.
- 기존 validation은 모델 선택/배정/weight에 이미 사용됐으므로 pilot로 해석한다.
- 정식 최종 검증은 새 holdout/session 또는 재설계한 전체 split에서 수행한다.
- nonexpert 0.70, hard-negative 규칙, backbone 크기·고정 여부, head 구조,
  학습 예산, 전처리, class order를 초기에는 통제한다.
- baseline R0에 hard-negative가 없고 R1에 있다면 배정 효과와 hard-negative 효과가 섞인다.
  초기에는 기존 R0/R1 전체 절차 비교라고 명시하고,
  정식 재분류 분석에서는 공통 hard-negative 규칙을 적용한 비교를 추가한다.
- 각 배정에서 줄어든 데이터로 실제 epoch당 update 수가 바뀔 수 있다.
  epoch 수뿐 아니라 update·표본 노출·학습 시간을 기록하고 예산을 맞춘다.
- test 성능으로 R2나 P*를 다시 선택하지 않는다.
- 실제 이벤트 지연은 onset과 실제 표시 시각으로 계산한다.
- 중첩음에서 단일 표시가 위험음 알림을 가리는지도 별도 평가한다.

## 7. 코드 연결 방식

### 7.1 기존 R0/R1

현재 기본 명령은 계속 사용한다.

~~~bash
python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --assignment predefined --ratio 0.70 \
  --out "$PILOT_PROJECT/pilot_R0"
~~~

정책 단계에서는 이 명령을 기준으로 사용한다. R1을 실제로 비교할 시점에는:

~~~bash
python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --assignment automatic --ratio 0.70 \
  --out "$PILOT_PROJECT/pilot_R1_retuned"
~~~

### 7.2 향후 R2

experiments/R2_custom_template.json은 **실행 불가 상태의 템플릿**이다.
현재 expert 목록을 시작 예시로 넣었을 뿐 선정된 R2가 아니다.
새 expert 배정을 확정하고 같은 14-class 모델을 재학습·예측 export·weight 생성한 뒤,
경로와 배정표를 수정하고 ready_for_evaluation을 true로 바꾼다.

~~~bash
python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --experiment-spec experiments/R2_selected.json \
  --out "$PILOT_PROJECT/pilot_R2_retuned"
~~~

custom spec은 데이터 설정으로만 읽으며 코드를 실행하지 않는다.

### 7.3 고정 정책 이전(T3)

~~~bash
python run_static_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --experiment-spec experiments/R2_selected.json \
  --policy-config-from "$PILOT_PROJECT/pilot_R0/policy_config.json" \
  --seed 42 \
  --out "$PILOT_PROJECT/pilot_R2_frozen"
~~~

새 R2 모델의 score와 weight를 사용하되 R0 cutoff를 그대로 가져온다.
class order와 seed, 새 형식의 evaluation sample digest가 다르면 중단한다.
이 옵션은 R0 모델/weight를 가져오지 않는다.

--policy-config-from을 빼면 같은 분할에서 해당 score로 O2를 다시 설정하여 T4를 구성한다.
T4의 추가 이득과 새로운 tuning 비용을 따로 기록한다.
GPU replay worker는 policy_config.json의 실험 경로를 읽으므로 기존 replay 명령을 그대로 쓴다.
각 실행은 별도 output 폴더를 사용하고 다른 실험의 manifest/cache를 섞지 않는다.

## 8. 근거 문헌과 적용 범위

- [HD-CNN, ICCV 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yan_HD-CNN_Hierarchical_Deep_ICCV_2015_paper.pdf):
  혼동 관계에서 전문 영역을 구성하는 원리. 이미지의 coarse/fine 구조 자체를 복제하지 않음.
- [Jung et al., IEEE Access 2020](https://pure.uos.ac.kr/en/publications/knowledge-distillation-in-acoustic-scene-classification/):
  혼동하는 음향 장면 쌍의 specialist 집중 학습 근거. 장면과 사건의 차이를 명시하고 증류는 필수로 도입하지 않음.
- [Standley et al., ICML 2020](https://proceedings.mlr.press/v119/standley20a.html):
  함께 학습할 때의 이득과 간섭을 실험하고 계산 예산 아래 묶음을 고르는 절차의 참고.
  클래스 배정과 다중 작업 학습이 같은 문제라고 주장하지 않음.
- [BEATs, ICML 2023](https://proceedings.mlr.press/v202/chen23ag.html):
  오프라인 공통 음향 특징 분석의 후보. 이 논문 자체가 현재 라벨의 최적 분할을 제공하지 않음.
- [Cances et al., WASPAA 2019](https://arxiv.org/pdf/1906.06909):
  라벨별 출력 파라미터 탐색 근거.
- [Pareto Testing, ICLR 2023](https://arxiv.org/abs/2210.07913):
  여러 성능·비용 기준에서 후보를 선택하고 독립 데이터로 검사하는 절차의 참고.

현재 수행하는 것은 정책 단계이며, 이 문서는 후속 재분류 실험의 사전 계획이다.
문헌은 설계 원리를 제공하며 현재 13개 라벨의 최적 배정·개선폭은 실험으로 검증한다.

