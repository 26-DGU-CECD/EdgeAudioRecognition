# Conda / requirements

현재 사용 환경은 `audio-policy`(정책 비교+PyTorch)와 `audio-yamnet`(TensorFlow) 두 개다. 별도 audio-torch 환경은 필수가 아니다. 이전 문서·requirements 주석에 audio-torch가 나오면 현재 팀 기준에서는 audio-policy로 읽는다.

## 1. 정책 비교만 재현

```powershell
conda create -n audio-policy python=3.11 pip -y
conda activate audio-policy
python -m pip install -r environments/requirements-policy.txt
python -c "import sys,numpy; print(sys.executable); print(numpy.__version__)"
```

## 2. EfficientAT / MobileNetV4 원음 추론 추가

```powershell
conda activate audio-policy
python -m pip install -r environments/requirements-torch-cuda128.txt
python -m pip install -r environments/requirements-torch.txt
python -m pip check
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda'))"
```

CUDA 전용 index의 Torch 3종을 먼저 설치하고, 일반 PyPI 의존성은 두 번째 requirements로 설치한다. 시스템 CUDA Toolkit 버전만으로 GPU 사용 여부를 판단하지 말고 위 GPU 실행 확인 결과를 본다.

## 3. YAMNet

```powershell
conda create -n audio-yamnet python=3.11 pip -y
conda activate audio-yamnet
python -m pip install -r environments/requirements-yamnet.txt
python -m pip check
```

이번 재현 조건은 Windows native TensorFlow CPU (`--device cpu`)다. 전체 모델 GPU 실험은 별도 WSL2/Linux 실행 조건으로 기록한다. 기존 단독 처리 시간 cache를 다른 장치의 측정값으로 해석하지 않는다.

## 설치 파일과 버전 기록

- `requirements-policy.txt`: NumPy 정책 비교.
- `requirements-common.txt`: 음원·평가 공통 의존성.
- `requirements-torch-cuda128.txt`: Torch 2.9.1 / torchvision 0.24.1 / torchaudio 2.9.1의 CUDA 12.8 wheel index.
- `requirements-torch.txt`: 공통 패키지, Torch 3종, timm.
- `requirements-yamnet.txt`: TensorFlow 2.20.0 / Hub 0.16.1 / tf-keras 2.20.1 / setuptools 81.0.0 및 공통 패키지.
- `requirements-yamnet-wsl-gpu.txt`: 별도 Linux GPU용 기존 설치 파일.
- `observed/*.packages.json`, `observed/*.freeze.txt`: 수집 시 설치되어 있던 두 환경의 실제 Python·패키지 버전. canonical requirements에 범위로 지정된 패키지의 당시 정확한 버전을 확인하는 기록이다.

observed freeze는 운영체제와 CUDA wheel suffix를 포함한 설치 기록이다. 다른 플랫폼에서 이를 그대로 설치하는 lockfile로 가정하지 않는다. 먼저 위 canonical 절차로 설치하고, 버전 차이가 결과에 영향을 주는지 기록한다.
