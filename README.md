# live_inference_refactored_independent

`realtime_inference.py`에 의존하지 않도록 기존 live inference 코드를 객체/파일 단위로 분리한 버전입니다.

## 실행

```bash
python3 main.py --efficientat-dir /workspace/EfficientAT
```

장치 목록 확인:

```bash
python3 main.py --list-devices
```

저음량 청크는 모델 추론까지 가지 않고 스킵:

```bash
python3 main.py --min-db 30 --skip-low-db
```

## 주요 구조

- `main.py`: 전체 객체 생성 및 실행
- `cli.py`: 실행 인자 처리
- `constants.py`: 상수 및 라벨 매핑
- `device_finder.py`: 입력 장치 탐색
- `microphone_module.py`: 마이크 스트림 관리
- `audio_queue.py`: 입력 콜백 데이터 큐
- `audio_buffer.py`: 청크 단위 버퍼링
- `audio_level_meter.py`: dBFS 계산
- `db_threshold_gate.py`: 임계값 판단
- `audio_preprocessor.py`: 작은 신호 감소 / 큰 신호 강조
- `efficientat_loader.py`: EfficientAT 모델 및 Mel frontend 로딩
- `label_mapper.py`: AudioSet 라벨을 사용자 정의 라벨로 매핑
- `model_inference.py`: 모델 입력 길이 조정 및 추론
- `audio_stream_controller.py`: 실시간 처리 루프 제어
