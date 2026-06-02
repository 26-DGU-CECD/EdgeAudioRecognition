# Edge Audio Refactor - no realtime_inference_ble_doa import

이 버전은 기존 통합 파일 `realtime_inference_ble_doa.py`를 import하지 않습니다.
업로드된 통합 파일 내부에 있던 역할을 다음처럼 나누었습니다.

- `main.py`: 객체 생성 및 전체 실행
- `microphone_module.py`: ReSpeaker 입력 스트림
- `audio_queue.py`: 입력 콜백 큐
- `audio_buffer.py`: 일정 길이 chunk 구성
- `audio_level_meter.py`: dBFS 계산
- `db_threshold_gate.py`: 임계값 판단
- `audio_preprocessor.py`: 작은 신호 감소 / 큰 신호 강조
- `model_inference.py`: EfficientAT 모델 로딩 및 추론
- `doa_usb_reader.py`: ReSpeaker USB DSP DOA
- `doa_audio_estimator.py`: raw mic TDOA 기반 DOA
- `doa_selector.py`: auto/audio/usb DOA 선택
- `ble_inference_server.py`: 앱 호환 BLE 알림
- `packet_builder.py`: 앱 전송 JSON 생성

## 실행

```bash
python3 main.py --efficientat-dir /workspace/EfficientAT
```

## 주의

이 프로젝트는 기존 `realtime_inference.py`와 `realtime_inference_ble.py`의 공통 함수/상수는 그대로 사용합니다.
하지만 `realtime_inference_ble_doa.py`는 필요하지 않습니다.
