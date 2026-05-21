# Jetson Nano EfficientAT Realtime Test

Jetson Nano + reSpeaker Mic Array v3.0 환경에서 EfficientAT pretrained 모델의 온디바이스 실시간 추론 가능성을 테스트한 코드입니다.

## 목적

- Jetson Nano에서 EfficientAT inference 가능 여부 확인
- reSpeaker Mic Array v3.0으로 실시간 오디오 입력 수집
- `mn04_as`, `mn05_as`, `mn10_as` 모델별 추론 시간 비교
- Jetson Nano에서는 학습하지 않고, 녹음/전처리/inference/classification만 수행

## 현재 Jetson 파일 구조

Host 기준:

```bash
~/efficientat_ws/
├── EfficientAT
├── audio
└── edge_audio_run
