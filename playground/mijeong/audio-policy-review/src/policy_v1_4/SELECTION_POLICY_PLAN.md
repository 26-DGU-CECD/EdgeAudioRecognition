# 전용 라벨 학습·무투표 최댓값 선택을 유지한 정책 실험 v1.2

사용자가 확인한 기존 최고 성능 방식은 모델별 전용 라벨을 다르게 학습하고,
각 담당 라벨의 점수를 모아 전체 최댓값 하나를 선택하는 방식이다.
이후의 주 기준 구조를 이 방식으로 정정한다. 모델 점수 평균·합산 voting은 필요하지 않다.

## 1. 코드 확인 결과와 아직 확정할 수 없는 부분

제공된 두 분할 폴더를 하나의 프로젝트로 확인했다.

| 확인한 파일 | 확인한 동작 | 해석 범위 |
|---|---|---|
| class_names.txt | background 포함 14개 클래스 | 제공된 학습·평가의 전역 클래스 목록 |
| evaluation_utils.py:164 | scores.argmax(axis=1), background 제외 코드 없음 | 개별 모델·결합 결과를 저장할 때 background도 후보 |
| ensemble_specialists.py:106–124 부근 | 각 모델 softmax → 클래스별 가중합 → 전체 클래스 정규화 | 제공된 F/G는 voting 구현이며 무투표 기준을 재현하지 않음 |
| specialist_data_utils.py | background 유지, expert/nonexpert 학습 표본 분리 | 제공된 specialist 학습 방식; 순수 전용 클래스 head라고 단정할 수 없음 |

**전용 라벨만 모은 뒤 최댓값을 선택하는 실제 실행 경로는 제공된 Python 소스에서 확인되지 않았다.**
따라서 개별 모델 평가에 background가 포함된다는 사실만으로 무투표 서비스의 background 처리까지 확정하지 않는다.
사용자에게 이를 기억해서 정하도록 요구하는 대신 아래 옵션으로 불확실성을 분리한다.
현재 제공된 checkpoint 중 어느 것이 사용자 최고 성능 실험인지도 확인하지 못했으므로,
기본 실행은 `supplied_cached_specialists_adaptation_reference`로 명시한다.
실제 최고 성능 checkpoint의 예측을 사용할 때는 별도 selection spec으로 경로·head 순서를 기록한다.

## 2. 보존할 점수 계산과 비교군

라벨 c의 담당 모델을 owner(c)라 할 때 s_c = p_owner(c)(c|x)로 둔다.
각 모델의 **실제 전체 head**에서 확률을 구한 뒤 담당 라벨만 가져온다.
담당 라벨만으로 softmax를 다시 계산하거나, 모은 점수의 합이 1이 되도록 정규화하지 않는다.
이렇게 모은 s는 한 개의 정규화된 확률분포가 아니며, 서로 다른 학습 라벨 공간이면
동일한 0.8이 같은 정확도를 의미한다고 보장할 수 없다.

| 비교군 | 단일 결과 선택 | 출력 확정 | 현재 구현 |
|---|---|---|---|
| S0 기준 | 담당 라벨 점수의 전체 최댓값 | 선택된 사건은 바로 출력 | 구현 |
| S1 공통 임계값 | S0와 동일 | 점수 ≥ 공통 τ일 때 출력, 아니면 미확정 | 구현 |
| S2 라벨별 임계값 | S0와 동일 | 점수 ≥ 해당 라벨 τ_c일 때 출력, 아니면 미확정 | 구현 |

S1/S2는 **확정·보류 정책**이며 최고 후보의 순위를 바꾸는 알고리즘은 아니다.
최고 후보가 탈락했다고 낮은 순위 후보를 대신 출력하지 않는다.
미확정은 background 예측과 구분하여 저장한다.
우선 W=2초·H=2초, checkpoint·전처리·배정을 고정한다.
정적 clip 점수 비교로 관찰 지연·GPU 추론 시간 감소를 주장하지 않는다.

## 3. background 처리의 민감도 확인

| 옵션 | 정의 | 주의 |
|---|---|---|
| global_max | 담당 사건 점수들과 모델별 background 점수 중 최댓값을 비교 | background 평균이 아닌 최대 선택. 기존 동작으로 확인된 것은 아님 |
| exclude | background를 최종 후보에서 제외 | S0는 사건 하나를 반드시 선택하므로 무음·배경 오경보 주의 |
| per_model_gate | 모델의 담당 사건 최고점수가 그 모델 background보다 클 때만 해당 모델 사건 후보 유지. 모두 제외되면 background | 다른 모델의 강한 background가 남은 모델을 전역 차단하지 않음 |

기본 `--background-mode all`은 3개 처리 가정 × 3개 확정 정책을 같은 데이터로 진단한다.
새로 9개 모델을 학습하는 실험이 아니다. 정책의 주 비교는 **같은 background 옵션 안에서** 한다.
처리 옵션 자체를 선택할 때도 development 결과만 사용하고 최종 test는 재사용하지 않는다.
head에 background 출력이 없으면 관련 옵션을 임의로 가정하지 않고 오류를 낸다.

## 4. 기준값 설정과 평가

validation의 recorded group을 seed=42 해시로 calibration/evaluation 양분한다.
동일 원본의 증강 clip이 두 집합에 걸치지 않는지 확인하고, 예측 파일의 전역 라벨을 metadata와 대조한다.
S1/S2 모두 동일한 목표 정밀도·최소 출력 수로 calibration subset에서만 τ를 선택한다.
초기 목표는 정밀도 95%, 최소 출력 20개이며 이는 실험 설정이지 보장이나 위험음 운영 기준이 아니다.
후보 τ grid는 0~0.995의 0.005 간격 및 0.999, 1.0이다.
조건을 만족하는 후보 중 correct accepted 수가 최대인 값을 고르고 동률이면 작은 τ를 택한다.
조건을 만족하지 못하면 null로 기록해 해당 정책/라벨은 출력 보류한다.
S1의 최소 표본은 전체 사건 출력, S2는 라벨별 출력에 적용되므로 희소 라벨은 별도 해석한다.

정밀도만 비교하지 않고 사건별 recall·F1, foreground macro F1,
foreground correct accept rate, background false accept rate, 미확정 수를 함께 본다.
확정을 줄여 정밀도가 올라간 것과 분류 자체가 개선된 것을 구분한다.
원본 학습·전문 배정이 validation을 사용했을 가능성이 있어 이 pilot은 탐색적 검증이다.
후속 본 실험에서는 독립 recorded-group 분할, 목표·후보 사전 고정, 독립 test 평가를 사용한다.
실제 onset 기준 지연·오경보/시간은 연속 음원과 onset/offset 주석이 확보된 후 평가한다.
미검출을 지연 집계에서 누락하지 않고 recall 및 검출 비율을 함께 제시한다.

## 5. 이후 선택 자체를 바꾸는 비교군

| 후속 방법 | 변경 내용 | 분리해서 검증할 이유 |
|---|---|---|
| 적격 후보 중 최댓값 | 모든 담당 라벨에 기준을 먼저 적용한 뒤 통과 후보 중 최댓값 | S2와 달리 1위 탈락 시 2위가 출력될 수 있음. S2 임계값의 정밀도는 이 정책에서 보장되지 않으므로 별도 재설정 필요 |
| 보정 후 최댓값 | 각 모델 점수 보정 후 담당 라벨의 전체 최댓값 | 모델 간 과신 편차를 줄이려는 방법. 현재 비학습 정책 실험과 구분하고 보정용 데이터 별도 확보 |
| 점수 차이 기준 | 최고점수 외에 1·2위 차이도 확정 조건에 추가 | 배경 거부·정밀도/recall을 같은 조건으로 확인. 담당 라벨 수·head 차이 영향을 받음 |
| 시간 연속 확인 | 동일 라벨을 연속 갱신에서 확인하거나 확정/해제 기준을 분리 | 현재 H2초에서 2회 연속 확인은 최단 추가 2초를 유발. 단발음 소실·2초 목표와의 충돌을 평가 |

정확한 전체 최댓값은 동일 window의 모든 specialist 후보가 준비되어야 확정할 수 있다.
먼저 도착한 결과를 즉시 출력하는 방식은 별도 비동기 선택 정책이며 전체 최댓값과 동등하지 않다.
기존 비동기 모델 실행은 유지하되 비교 대상의 window ID·시점을 일치시킨다.
이 버전은 저장된 동기 window 점수의 선택 정책만 구현하며, live 비동기 작업 스케줄러는 구현하지 않았다.

## 6. 논문에서 가져온 부분과 적용 한계

- **Geifman & El-Yaniv, NeurIPS 2017, Selective Classification for Deep Neural Networks.**
  [원문 페이지](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
  신뢰도가 부족한 예측을 거부하여 오류와 coverage를 조절하는 연구다.
  S1/S2에는 확정·보류와 coverage 동시 평가 원칙을 적용한다.
  본 pilot의 단순 validation grid search는 논문의 위험 보장 절차를 재현하지 않는다.
- **Cances et al., WASPAA 2019, Evaluation of post-processing algorithms for polyphonic sound event detection.**
  [원문](https://arxiv.org/abs/1906.06909)
  SED의 시간별 예측에 smoothing·thresholding을 적용하고 class-dependent 설정을 비교한다.
  S2에는 클래스별 기준을 분리하는 실험 원칙을 적용한다.
  현재 출력은 2초 clip 점수이므로 논문의 프레임별 event segmentation을 그대로 구현한 것은 아니다.
- **Guo et al., ICML 2017, On Calibration of Modern Neural Networks.**
  [원문 페이지](https://proceedings.mlr.press/v70/guo17a.html)
  신경망 confidence와 실제 정답률의 차이를 분석하고 temperature scaling을 평가한다.
  후속 보정 후 max의 근거이며 현 S0/S1/S2에는 보정을 적용하지 않는다.
  원 논문은 서로 다른 전용 라벨 공간의 specialist 확률을 직접 비교해도 된다는 증거가 아니다.
  전용 subset head의 조건부 확률에서는 temperature scaling만으로 전역 비교 문제가 해결된다고 가정하지 않는다.

## 7. 실행

기존 가중 voting 경로는 참고 비교로 남겨 두고 새 진입점을 사용한다.
원본 소스·checkpoint는 변경하지 않는다. 기존 GPU 환경에서도 NumPy만 있으면 저장 예측으로 먼저 실행 가능하다.

```bash
python run_selection_pilot.py \
  --project-root /home/plass/YAMNet/audio_experiments_DEF \
  --background-mode all \
  --out selection_reference
```

기본값 predefined/0.70의 기존 저장 예측을 재사용한다. **사용자 최고 성능 run의 재현이라고 해석하지 않는다.**
실제 최고 성능 run의 저장 예측은 selection_spec.example.json을 복사·수정하고
head의 정확한 라벨 순서와 파일 경로를 검증한 뒤 ready_for_evaluation=true로 바꾼다.

```bash
python run_selection_pilot.py \
  --project-root /home/plass/YAMNet/audio_experiments_DEF \
  --selection-spec my_best_selection.json \
  --background-mode all \
  --out selection_best_run
```

NPZ는 paths, y_true, logits(또는 probabilities)를 가진다.
y_true는 각 모델의 로컬 head ID가 아닌 class_names.txt의 **전역 ID**여야 한다.
모델별 head 크기·순서는 서로 달라도 되며, 라벨 재배정 시 expert_classes와 새 예측 경로를 바꾼다.
출력은 selection_comparison.csv, selection_per_class.csv, selection_config.json이다.
설정에는 파일 hash, ownership, head 순서, 기준값, background 모드, 분할 수, 비정규화 여부를 기록한다.
기존 가중 fusion용 policy_config.json과 호환되는 것으로 취급하지 않는다.

검증: `python -m unittest discover -p 'test_*.py' -v`

## 8. 이번 참고 실행 결과

제공된 predefined/0.70 모델의 val 저장 예측에서 calibration 11,965개,
evaluation 11,882개를 사용했다. recorded-group 중복은 0개다.
**실제 최고 성능 run의 checkpoint 또는 기존 무투표 실행 경로를 확인한 결과는 아니다.**
이 실행은 softmax_logits 입력으로 가정했으며 원래 서비스가 모델 간 raw logits를 비교했다면
동일한 기준군이 아니다. 선택 점수의 변환 방식도 실제 서비스 소스 확보 후 대조해야 한다.

background=global_max 가정 안에서의 결과:

| 정책 | 출력 정밀도 | 사건 정답 출력률 | 배경 오출력률 | foreground macro F1 |
|---|---:|---:|---:|---:|
| S0 | 82.30% | 79.00% | 14.56% | 79.40% |
| S1 | 95.39% | 56.28% | 4.38% | 62.97% |
| S2 | 95.08% | 56.37% | 4.20% | 60.83% |

사건 정답 출력률은 (사건을 올바르게 출력한 수)/(모든 실제 foreground clip 수)다.
배경 오출력률은 (사건을 출력한 background clip 수)/(모든 실제 background clip 수)다.
S2가 항상 우수하지 않으며 정밀도 향상과 누락 증가의 trade-off를 확인했다.
S0의 background=exclude 가정에서는 모든 background clip에 사건 라벨을 출력해
배경 오출력률이 100%였다. 이를 기존 서비스의 실제 성능이라고 해석하지 않는다.
23개 코드 테스트가 통과했으며, GPU 신규 추론·실시간 지연·실제 onset 평가를 수행한 것은 아니다.
