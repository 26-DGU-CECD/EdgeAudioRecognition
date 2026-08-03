# Budget60 threshold 산출 및 검증 근거

## 1. 문서 목적

이 문서는 [`thresholds_budget60.json`](./thresholds_budget60.json)의 산출 목적과
검증 결과를 기록한다.

해당 JSON은 전체 순수 헛알림을 시간당 60회 이하로 제한하면서 안전 관련 클래스의
recall을 우선하는 후보 설정이다. 기본 설정인 `thresholds.json`을 아직 대체하지
않으며, 실제 Raspberry Pi와 하우징 조건에서 검증한 뒤 적용 여부를 결정한다.

## 2. 최적화 조건

- 튜닝 데이터: validation set
- 목표: 윈도우 단위 결과를 합친 순수 헛알림 `60/h` 이하
- 안전 클래스 가중치: `3.0`
- 그 외 클래스 가중치: `1.0`
- 안전 클래스:
  - `siren`
  - `scream`
  - `crying`
  - `baby_cry`
  - `gunshot`
  - `glass_break`
- 각 클래스는 자신의 sigmoid 확률과 threshold만 비교한다.
- 서로 다른 검출기의 확률을 직접 비교하거나 투표하지 않는다.

## 3. Budget60 클래스별 결과

아래 결과는 batch 64 점수로 최적화한 `thresholds_budget60.json`을 실제 Pi 추론
경로와 같은 batch 1 전처리로 측정한 값이다.

| 클래스 | threshold | precision | recall | FP/h | 기존 F1 recall | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| siren | 0.99993980 | 1.000 | 0.500 | 0.1 | 0.949 | -0.450 |
| scream | 0.50764245 | 0.834 | 0.758 | 17.7 | 0.751 | +0.007 |
| crying | 0.98959190 | 0.975 | 0.172 | 1.3 | 0.781 | -0.609 |
| bicycle_bell | 0.99085050 | 0.972 | 0.781 | 3.4 | 0.972 | -0.192 |
| baby_cry | 0.94649970 | 0.934 | 0.960 | 8.6 | 0.967 | -0.007 |
| gunshot | 0.86728835 | 0.945 | 0.945 | 11.3 | 0.966 | -0.021 |
| glass_break | 0.99963117 | 1.000 | 0.697 | 0.2 | 0.954 | -0.257 |
| water | 0.99999774 | 0.952 | 0.400 | 5.2 | 0.744 | -0.344 |
| knock | 0.98402023 | 0.645 | 0.819 | 7.8 | 0.708 | +0.111 |
| dog | 0.99917680 | 0.886 | 0.733 | 4.5 | 0.750 | -0.017 |
| cat | 0.99999607 | 0.910 | 0.505 | 7.6 | 0.770 | -0.265 |

전체 결과:

- 순수 헛알림: `61.7/h` (`610`개 윈도우)
- 클래스별 헛알림 단순합: `67.7/h`
- 평균 recall: `0.661`
- 기존 F1 설정을 같은 경로로 측정한 결과: `223.1/h`, 평균 recall `0.847`

클래스별 FP/h의 단순합과 순수 헛알림이 다른 이유는 한 윈도우에서 여러 클래스가
동시에 검출될 수 있기 때문이다.

## 4. Batch 64와 Pi batch 1 드리프트

| 검출기 | 클래스 | 최대 차이 | 평균 차이 | 판정 뒤집힘 |
|---|---|---:|---:|---:|
| YAMNet | siren / scream / crying | 1e-6 | 약 0 | 1 / 0 / 1 |
| EfficientAT | 담당 4개 클래스 | 3e-6 | 약 0 | 클래스별 0~1 |
| MobileNetV4 | water | 0.9966 | 0.0066 | 153 |
| MobileNetV4 | cat | 0.9951 | 0.0074 | 84 |
| MobileNetV4 | knock | 0.9731 | 0.0015 | 29 |
| MobileNetV4 | dog | 0.9988 | 0.0020 | 17 |

YAMNet과 EfficientAT은 batch 크기에 따른 차이가 사실상 없다. 기존 MobileNetV4
전처리는 3차원 mel tensor를 `AmplitudeToDB(top_db=80)`에 전달하여, batch 전체가
하나의 기준 최댓값을 공유했다. 이 때문에 35,565개 클립 중 283개의 판정이
batch 64와 batch 1 사이에서 뒤집혔다.

현재 `MelImageTransform`은 `AmplitudeToDB` 호출 전에 명시적인 channel 축을
추가한다. 따라서 각 샘플이 독립적인 `top_db` 기준을 사용한다.

- 기존 Pi batch 1 출력은 그대로 유지된다.
- batch 2 이상에서도 다른 샘플의 음량이 현재 샘플에 영향을 주지 않는다.
- 이후 threshold 튜닝은 수정된 전처리 또는 Pi와 동일한 batch 1 경로에서 수행해야 한다.

Pi 경로 점수로 다시 최적화한 실험에서는 순수 헛알림 `59.9/h`가 보고되었다.
변경된 클래스는 `scream`, `glass_break`, `dog`, `knock` 등이며 변화 폭은 작았다.
하지만 정확한 전체 threshold JSON이 현재 저장소에 없으므로 이 수치는 참고 결과로만
기록하며 `thresholds_budget60.json`에는 적용하지 않는다.

## 5. 하우징 조건별 결과

동일한 12,000개 클립에 clean, mild, moderate 하우징 조건을 적용한 recall이다.

| 클래스 | clean | mild | moderate |
|---|---:|---:|---:|
| siren | 0.509 | 0.152 | 0.010 |
| bicycle_bell | 0.764 | 0.088 | 0.000 |
| glass_break | 0.680 | 0.278 | 0.001 |
| gunshot | 0.937 | 0.790 | 0.486 |
| baby_cry | 0.954 | 0.895 | 0.716 |
| scream | 0.754 | 0.715 | 0.533 |
| crying | 0.172 | 0.112 | 0.074 |
| water | 0.375 | 0.369 | 0.231 |
| cat | 0.502 | 0.417 | 0.417 |
| knock | 0.841 | 0.889 | 0.937 |
| dog | 0.698 | 0.698 | 0.718 |

조건별 전체 순수 헛알림은 다음과 같다.

```text
clean 64.5/h → mild 58.5/h → moderate 92.4/h
```

moderate 조건에서는 `siren`, `bicycle_bell`, `glass_break`의 recall이 사실상
0에 가까워지지만 헛알림은 오히려 증가한다. 고주파 성분이 크게 감쇠되고 뭉개진
잡음이 다른 소리로 해석되기 때문이다.

이는 threshold만 조정해서 해결할 수 있는 문제가 아니다. 세 클래스를 유지하려면
마이크 하우징에 음향 포트를 두고, 실제 조립 상태에서 다시 측정해야 한다.

`knock`, `dog`, `cat`은 감쇠에 상대적으로 덜 민감하다. MobileNetV4 전처리의
샘플별 min-max 정규화가 전체 음량 감쇠를 일부 상쇄하기 때문이다.

## 6. 검출기 한계

60/h 예산에서는 특히 다음 클래스의 recall 손실이 크다.

- `siren`: `0.949 → 0.500`
- `crying`: `0.781 → 0.172`

안전 클래스에 3배 가중치를 주어도 두 클래스의 오탐 단가가 높아 threshold가 크게
상승했다. 이는 threshold 선택만의 문제가 아니라 현재 검출기의 클래스 분리력
한계다. 두 클래스의 recall을 복구하려면 추가 데이터, hard-negative 학습 또는
별도 검출기 개선이 필요하다.

## 7. 적용 방법과 주의사항

저장소를 `jetson_backend` 디렉터리 기준으로 실행할 때 후보 threshold를 지정한다.

```bash
./venv/bin/python main.py \
  --thresholds edge_audio_pi/parallel_detectors/report/thresholds_budget60.json
```

운영 적용 전에는 다음을 확인한다.

1. 실제 Raspberry Pi, reSpeaker 마이크, 최종 하우징 조합으로 측정한다.
2. 일상 환경의 순수 음성·생활 소음을 충분히 포함해 FP/h를 다시 계산한다.
3. `siren`, `crying`의 낮은 recall을 안전 요구사항과 비교한다.
4. `siren`, `bicycle_bell`, `glass_break`용 음향 포트를 적용한 뒤 재평가한다.
5. 검증이 끝나기 전에는 기본 `thresholds.json`을 교체하지 않는다.
