# Jetson Nano EfficientAT Realtime Test

## 목적

Jetson Nano + reSpeaker Mic Array v3.0 환경에서 EfficientAT pretrained 모델의 온디바이스 실시간 추론 가능성을 테스트한다.

## 모델

- mn04_as: Jetson Nano 실시간 테스트용 우선 후보
- mn05_as: mn04_as보다 약간 무겁지만 성능 보완 후보
- mn10_as: 기존 테스트 모델이나 Jetson Nano에서 느려 실시간 테스트에서는 후순위

## 실행 환경

- Jetson Nano
- Docker: nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
- reSpeaker Mic Array v3.0
- PyTorch CUDA inference

## Docker 실행 예시

```bash
sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOME/efficientat_ws:/workspace" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

:wq
wq
exit
;
