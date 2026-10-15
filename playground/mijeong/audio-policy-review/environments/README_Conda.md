# 오디오 정책 실험용 Conda 환경

작성일: 2026-10-16. 대상: Windows PC + VS Code, RTX 4090.

Conda로 Python과 pip를 준비하고, 각 환경 안에서 `python -m pip`로 패키지를 설치합니다.
Python은 새 환경에서 3.11로 통일합니다. 원본 프로젝트의 requirements와 모델 파일은 변경하지 않았습니다.
이 폴더의 설치 파일을 함께 유지하세요. `-r`로 연결된 파일은 해당 requirements 파일 위치를 기준으로 읽습니다.

## 환경 선택

| 환경 | 실행할 작업 | Windows native GPU |
|---|---|---|
| audio-policy | 저장 NPZ 기반 S0/S1/S2 및 정적 voting 비교 | GPU 불필요 |
| audio-torch | EfficientAT, MobileNetV4 추론·학습 | CUDA 12.8 빌드 사용 |
| audio-yamnet | YAMNet embedding 추출·head 추론·학습 | TensorFlow 2.20은 CPU |
| audio-yamnet-gpu | 같은 PC의 WSL2에서 YAMNet GPU 실행 | WSL2 내부 CUDA 사용 |

우선 audio-policy만 만들어 정적 실험을 진행하면 됩니다. 신규 원음 추론이 필요할 때 나머지 환경을 만드세요.
PyTorch와 TensorFlow를 한 환경에 함께 설치하지 않습니다.

## 0. VS Code 터미널 준비

아래 Windows 명령은 PowerShell용입니다. 먼저 `conda --version`이 동작하는지 확인하세요.
Conda가 인식되지 않으면 Anaconda PowerShell Prompt에서 실행하거나, Anaconda Prompt에서
`conda init powershell`을 한 번 실행한 뒤 VS Code를 완전히 닫았다가 다시 여세요.

현재 파일 위치 그대로 사용하는 경우:

```powershell
Set-Location "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\conda-setup"
```

파일을 다른 위치로 복사했다면 해당 폴더로 이동합니다.

## 1. 지금 필요한 정적 정책 실험 환경

```powershell
conda create -n audio-policy python=3.11 pip -y
conda activate audio-policy
python -m pip install -r requirements-policy.txt
python -m pip check
python -c "import sys,numpy; print(sys.executable); print(sys.version); print('NumPy',numpy.__version__)"
```

설치 후 다음과 같이 v1.2를 확인할 수 있습니다.

```powershell
$pilotDir = "C:\Users\PLASS\Desktop\audio-policy-review-20261015T164106Z-1-001\audio-policy-review\policy-pilot"
$audioProject = "C:\Users\PLASS\Desktop\audio_experiments_DEF"
Set-Location $pilotDir
python -m unittest discover -p "test_*.py" -v
python run_selection_pilot.py --project-root "$audioProject" --assignment predefined --ratio 0.70 --background-mode all --seed 42 --target-precision 0.95 --minimum-accepted 20 --out "$audioProject\policy_results\S01_conda_reference"
```

이 단계는 저장된 예측만 사용하므로 GPU와 원음이 필요하지 않습니다.
정적 평가에서는 `config.py`나 metadata의 Linux 경로를 변경할 필요가 없습니다.

## 2. EfficientAT / MobileNetV4: Windows GPU 환경

아래 명령 전에는 설치 파일 폴더로 다시 이동하세요.

```powershell
Set-Location "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\conda-setup"
conda create -n audio-torch python=3.11 pip -y
conda activate audio-torch
python -m pip install -r requirements-torch-cuda128.txt
python -m pip install -r requirements-torch.txt
python -m pip check
python -c "import torch,torchaudio,torchvision,timm; print('torch',torch.__version__); print('torchaudio',torchaudio.__version__); print('torchvision',torchvision.__version__); print('CUDA runtime',torch.version.cuda); print('CUDA available',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA GPU not detected'; print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda'))"
```

PyTorch 세 패키지는 원본의 2.9.1 / 0.24.1 / 2.9.1 조합을 유지합니다.
CUDA wheel은 공식 cu128 인덱스에서 먼저 설치합니다. 이어서 공통 패키지와 timm을 일반 PyPI에서 설치합니다.
해당 wheel에는 필요한 CUDA runtime 의존성이 포함되므로, 이 프로젝트의 일반 추론·학습을 위해
별도 `conda install cudatoolkit`을 추가할 필요는 없습니다. NVIDIA 드라이버는 필요합니다.

## 3. YAMNet: Windows native CPU 환경

```powershell
Set-Location "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\conda-setup"
conda create -n audio-yamnet python=3.11 pip -y
conda activate audio-yamnet
python -m pip install -r requirements-yamnet.txt
python -m pip check
python -c "import tensorflow as tf,tensorflow_hub as hub,tf_keras; print('TensorFlow',tf.__version__); print('Hub',hub.__version__); print('tf-keras',tf_keras.__version__); print('GPU',tf.config.list_physical_devices('GPU')); print(tf.reduce_sum(tf.ones([2,2])).numpy())"
```

Windows native에서는 GPU 목록이 빈 배열인 것이 예상 결과입니다.
기존 replay exporter를 YAMNet CPU 기능 점검에 사용할 때는 `--device cpu`를 지정합니다.
CPU 결과의 시간을 세 모델 GPU 동시 운용 시간으로 해석하지 마세요.

TensorFlow 2.20.0, Hub 0.16.1, setuptools 81.0.0은 원본 버전을 유지했습니다.
Hub가 간접 설치하는 tf-keras는 2.20.1로 명시해 최신 2.21 계열로 선택되는 것을 막았습니다.
setuptools 82부터는 Hub 0.16.1이 사용하는 pkg_resources가 제거되므로 setuptools를 임의로 올리지 마세요.
tf-keras 설치는 Hub의 호환성 의존성을 충족하기 위한 것입니다. 기존 `tf.keras` head를 변경하거나
`TF_USE_LEGACY_KERAS=1`을 일괄 설정하지 않습니다. 실제 best.keras 로드 확인은 별도로 필요합니다.

## 4. 선택 사항: 같은 PC의 WSL2에서 YAMNet GPU

WSL2 Ubuntu, NVIDIA GPU 접근, WSL 내부 Conda가 준비된 경우에만 다음 Bash 명령을 실행합니다.
Windows Conda 환경과 WSL Conda 환경은 별도입니다. 현재 확인 시점에는 설치된 WSL 배포판이 없었습니다.

```bash
cd /mnt/c/Users/PLASS/Documents/Codex/2026-10-16/r/outputs/conda-setup
nvidia-smi
conda create -n audio-yamnet-gpu python=3.11 pip -y
conda activate audio-yamnet-gpu
python -m pip install -r requirements-yamnet-wsl-gpu.txt
python -m pip check
python -c "import tensorflow as tf,tensorflow_hub as hub,tf_keras; print(tf.__version__); print(hub.__version__); print(tf_keras.__version__); g=tf.config.list_physical_devices('GPU'); print(g); assert g, 'TensorFlow GPU not detected'"
```

TensorFlow 공식 CUDA GPU 지원은 Windows native에서 2.10까지이고, 현재 2.20 구성의 GPU 실행에는 WSL2/Linux를 사용합니다.
GPU를 위해 TensorFlow를 2.10으로 내리면 기존 .keras checkpoint 호환성도 다시 확인해야 하므로 이 설치안에는 포함하지 않았습니다.

## VS Code와 환경 기록

VS Code에서 `Ctrl+Shift+P` → `Python: Select Interpreter`로 작업에 맞는 환경을 선택한 뒤 새 터미널을 여세요.
`python -c "import sys; print(sys.executable)"`로 터미널의 실제 Python 경로를 확인합니다.
현재 PC의 기본 PATH에는 Python 2.7이 있으므로, 활성화되지 않은 터미널에서 그대로 실행하지 마세요.

설치와 import/GPU 확인이 통과하면 활성 환경의 패키지 목록을 기록합니다.

```powershell
python -m pip freeze | Set-Content -Encoding UTF8 "requirements-installed-$env:CONDA_DEFAULT_ENV.txt"
```

Conda 환경 생성·실제 모델 로드·GPU 추론을 이 파일 작성 과정에서 새로 실행하지는 않았습니다.
이 파일들은 설치 입력이며, 과거 학습 환경의 전체 lock 파일을 복원한 결과는 아닙니다.
공통 패키지 범위는 원본을 유지했고 NumPy만 검증에 사용한 1.26.4로 고정했습니다.
같은 버전이라도 Windows CPU/GPU·Linux 간 처리시간과 수치가 완전히 같다고 가정하지 않습니다.
원음 경로 연결과 무투표 replay 확장은 requirements 설치와 별도로 진행해야 합니다.

## 붙여넣은 연구 방향과 환경의 연결

소리 onset부터 최초 올바른 표시까지 2초 이내라는 목표를 유지합니다.
모델 연산 시간과 입력 관찰·큐·확정 대기를 분리해서 기록합니다.

| 실험 범위 | 사용할 환경 | 추가 의존성 판단 |
|---|---|---|
| 저장 점수 S0/S1/S2 | audio-policy | NumPy만 필요 |
| B0/B1 및 P1/P3의 새 음원 평가 | audio-torch + audio-yamnet | 기존 모델별 프레임워크 사용 |
| 짧은·긴 문맥 및 부분 구간 학습 M1/M2 | 해당 모델 환경 | 현재 구성으로 구현 시작 가능; 구간 라벨·전처리 검증 필요 |
| 중간 분류기와 내부 early exit P2/M3 | 해당 모델 환경 | 일반 중간 출구는 기존 프레임워크로 구현 가능 |
| 전문 라벨 재배정·거부 강화 M4/M5 | 해당 모델 환경 | 기존 학습 도구 사용; 추가 데이터와 비교 설계 필요 |
| 관찰 정지 보정 P4, HypEE 등 논문별 재현 | 구현 결정 후 별도 환경 검토 | 현재 requirements가 해당 논문 구현까지 지원한다고 보장하지 않음 |

진행 순서는 기준 선택 규칙 고정 → 정적 출력 정책 → 무투표 replay 연결 → 갱신 간격 단축 →
관찰·확정 규칙 → 짧은·긴 문맥 비교 → 내부 early exit/라벨 재배정의 개별 비교 → 유효한 방법 결합으로 제안합니다.
짧은 문맥에는 사건 구간 검증이 필요하며, 기존 2초 요약 임베딩을 나누는 것으로 대체할 수 없습니다.
계산량 감소는 실제 깊은층 실행을 생략했을 때 따로 측정합니다.
현재 v1.2의 selection 출력과 가중 voting replay 사이의 연결 수정은 패키지 설치만으로 해결되지 않습니다.
이 연구 계획의 논문별 주장과 수치 전체를 이번 requirements 정리에서 재검증한 것은 아닙니다.

## 확인한 공식 자료

- [PyTorch 2.9.1 공식 설치 조합](https://pytorch.org/get-started/previous-versions/#v291)
- [TensorFlow Windows/WSL2 설치 안내](https://www.tensorflow.org/install/pip)
- [TensorFlow Hub 0.16.1 초기화 코드](https://github.com/tensorflow/hub/blob/v0.16.1/tensorflow_hub/__init__.py)
- [tf-keras 2.20.1 배포 정보](https://pypi.org/project/tf-keras/2.20.1/)
- [setuptools 82.0.0 변경 기록](https://setuptools.pypa.io/en/latest/history.html#v82-0-0)
