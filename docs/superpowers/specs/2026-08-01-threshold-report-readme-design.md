# Threshold Report README 설계

## 목표

`jetson_backend/edge_audio_pi/parallel_detectors/report`에 있는 threshold 파일들의
목적과 차이, 적용 방법, 운영상 주의사항을 한 문서에서 확인할 수 있게 한다.

## 산출물

- 파일: `jetson_backend/edge_audio_pi/parallel_detectors/report/README.md`
- 독자: threshold 후보를 선택하거나 Raspberry Pi 실행 옵션을 구성하는 개발자
- 범위: 현재 저장소에 있는 네 가지 threshold JSON과 관련 calibration 파일

## 문서 구성

1. threshold가 클래스별 sigmoid 확률에 독립적으로 적용된다는 판정 방식을 설명한다.
2. 다음 파일을 비교표로 정리한다.
   - `thresholds.json`: calibration ID가 포함된 기본 F1 설정
   - `thresholds_budget60.json`: 전체 순수 헛알림을 약 60회/시간으로 제한하려는 후보
   - `thresholds_conservative.json`: 클래스별 헛알림 20회/시간 제약을 목표로 한 후보
   - `thresholds_enclosure.json`: 하우징 감쇠를 고려해 완화한 60회/시간 후보
3. 기본 운영 설정으로 `thresholds.json`을 권장한다. 나머지 세 파일은 실제 Pi,
   마이크, 최종 하우징 조건에서 재검증하기 전까지 실험 후보로 명시한다.
4. `--thresholds` 옵션을 사용한 실행 예시와 기본 파일로 되돌리는 방법을 제시한다.
5. threshold payload의 calibration ID와 detector calibration ID가 다르면 기동이
   실패하며, calibration 필드가 없는 구형 후보 파일은 경고 후 실행된다는 현재 동작을
   설명한다.
6. Budget60의 상세 수치와 실험 근거는 `thresholds_budget60.md`로 연결한다.

## 정확성 기준

- 파일명, JSON 메타데이터, CLI 옵션은 현재 저장소 내용과 일치해야 한다.
- 검증되지 않은 후보를 운영 권장 설정으로 표현하지 않는다.
- FP/h 목표와 실제 측정 결과를 혼동하지 않는다.
- 문서 링크는 `report/README.md` 기준 상대 경로로 작성한다.

## 검증

- 네 JSON의 메타데이터와 README 비교표를 수동 대조한다.
- `rg`로 `--thresholds` 옵션과 calibration 검증 동작을 코드에서 재확인한다.
- `git diff --check`로 Markdown 공백 오류를 확인한다.
