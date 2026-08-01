# Threshold 설정 안내

이 디렉터리에는 병렬 소리 검출기의 클래스별 threshold와 확률 calibration 정보가
있다. 운영 기본값은 [`thresholds.json`](./thresholds.json)이다. 다른 threshold
파일은 검증 목적의 후보이므로 실제 Raspberry Pi, reSpeaker 마이크, 최종 하우징
조건에서 다시 측정한 뒤 사용해야 한다.

## 판정 방식

각 검출기는 담당 클래스의 sigmoid 확률을 출력한다. 클래스는 자신의 확률이 자신의
threshold 이상일 때만 검출 후보가 된다.

```text
클래스 검출 조건: probability >= threshold
판정 여유:       margin = probability - threshold
```

클래스끼리 확률을 직접 비교하거나 검출기 간 투표를 하지 않는다. 여러 클래스가 동시에
threshold를 넘으면 모두 후보가 되며, 대표 결과는 `margin`이 가장 큰 클래스다.

일반적으로 threshold를 낮추면 recall과 헛알림이 함께 증가하고, 높이면 헛알림과
recall이 함께 감소한다. 따라서 파일 이름이나 목표 수치만 보고 운영 설정을 바꾸지 말고
실제 사용 환경의 recall과 시간당 헛알림(FP/h)을 함께 확인해야 한다.

## 파일 선택

| 파일 | 튜닝 목적 | calibration ID | 권장 용도 |
|---|---|---|---|
| [`thresholds.json`](./thresholds.json) | validation set의 클래스별 F1 최대화 | 있음 | 현재 기본 운영 설정 |
| [`thresholds_budget60.json`](./thresholds_budget60.json) | 전체 순수 헛알림 약 60회/시간 이하를 목표로 안전 클래스 recall 가중 | 없음 | 헛알림 예산 실험 후보 |
| [`thresholds_conservative.json`](./thresholds_conservative.json) | 클래스별 FP 20회/시간 이하에서 recall 최대화 | 없음 | 하우징 감쇠 대비 실험 후보 |
| [`thresholds_enclosure.json`](./thresholds_enclosure.json) | 클래스별 FP 60회/시간 이하에서 recall 최대화 | 없음 | 더 완화된 하우징 실험 후보 |
| [`calibration.json`](./calibration.json) | detector별 logit을 threshold 튜닝에 사용한 확률 스케일로 보정 | detector별 ID 포함 | 모델과 함께 배포하는 필수 파일 |

### 기본 설정: `thresholds.json`

현재 실행 경로의 기본 파일이다. 세 검출기의 calibration ID를 포함하므로 threshold를
튜닝한 확률 스케일과 현재 detector의 확률 스케일이 같은지 기동 시 확인할 수 있다.

### Budget60 후보

안전 관련 클래스에 3배 가중치를 주면서 전체 순수 헛알림을 60회/시간 이하로 제한하려
만든 후보이다. Pi와 같은 batch 1 경로에서 재측정한 결과는 `61.7/h`였으며 목표와 실제
측정 결과가 정확히 같지는 않다. `siren`, `crying` 등의 recall 손실과 하우징 조건별
결과를 포함한 상세 근거는
[`thresholds_budget60.md`](./thresholds_budget60.md)를 참고한다.

### 하우징 후보

`thresholds_conservative.json`과 `thresholds_enclosure.json`은 3D 프린팅 하우징의
음향 감쇠를 고려해 기본 F1 threshold보다 낮거나 같은 값으로 완화한 후보이다.
실제 하우징 녹음으로 재검증되지 않았으며, 특히 고주파 성분에 민감한 클래스는
threshold 완화만으로 성능을 복구하지 못할 수 있다.

## 적용 방법

`jetson_backend` 디렉터리에서 `--thresholds` 옵션으로 사용할 파일을 지정한다.

```bash
./venv/bin/python main.py \
  --thresholds edge_audio_pi/parallel_detectors/report/thresholds_budget60.json
```

기본 설정으로 실행하려면 옵션을 생략한다.

```bash
./venv/bin/python main.py
```

환경 변수로 지정할 때는 `EDGEAUDIO_THRESHOLDS`에 파일 경로를 설정할 수 있다.
명령행의 `--thresholds` 값이 있으면 해당 값이 적용된다.

## Calibration 호환성

threshold는 calibration된 확률 스케일에 종속된다. 같은 숫자라도 calibration이나
checkpoint가 달라지면 같은 판정을 보장하지 않는다.

- threshold JSON에 `calibration` ID가 있고 현재 detector의 ID와 다르면 기동을
  중단한다.
- `thresholds.json`에는 현재 세 detector의 calibration ID가 기록되어 있다.
- 나머지 후보 파일에는 calibration ID가 없어 실행 시 경고만 출력된다. 이 경우
  프로그램은 계속 실행되지만 스케일 호환성을 자동으로 확인할 수 없다.
- `calibration.json`, detector checkpoint, threshold JSON은 같은 평가 결과에서 나온
  조합으로 배포해야 한다.

후보 파일을 운영에 적용하려면 현재 calibration 스케일에서 threshold를 다시 튜닝하고
JSON에 detector별 calibration ID를 기록하는 것이 안전하다.

## 운영 적용 전 확인 사항

1. 실제 Raspberry Pi, reSpeaker 마이크, 최종 하우징을 사용한다.
2. 목표 소리뿐 아니라 음성, TV, 음악, 생활 소음 등 실제 음성·비음성 배경을 포함한다.
3. 클래스별 recall과 FP/h, 전체 순수 FP/h를 각각 측정한다.
4. `siren`, `scream`, `crying`, `baby_cry`, `gunshot`, `glass_break` 등 안전 관련
   클래스의 recall 저하가 허용 범위인지 확인한다.
5. 검증 결과와 calibration ID를 함께 기록한 뒤 기본 설정 교체 여부를 결정한다.
