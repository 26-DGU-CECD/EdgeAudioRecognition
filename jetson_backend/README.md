# Raspberry Pi 병렬 전문 소리 검출 백엔드

Raspberry Pi 5와 Seeed ReSpeaker Mic Array v3.0에서 세 개의 전문 검출기를
동시에 실행하고, 감지 결과를 기존 BLE GATT 형식으로 전송한다.

## 빠른 설치 및 실행

아래 명령은 Raspberry Pi에서 실행한다.

```bash
# 1. 시스템 패키지 설치
sudo apt update
sudo apt install -y \
  git python3-dev python3-venv \
  portaudio19-dev libsndfile1 \
  bluez python3-dbus python3-gi \
  libgirepository1.0-dev

# 2. 추론 브랜치만 clone
git clone \
  --branch ensemble-inference \
  --single-branch \
  https://github.com/26-DGU-CECD/EdgeAudioRecognition.git

cd EdgeAudioRecognition/jetson_backend

# 3. Python 가상환경 생성
# BLE의 dbus/gi apt 패키지를 venv에서도 사용해야 한다.
python3 -m venv --system-site-packages venv
./venv/bin/python -m pip install --upgrade pip wheel
./venv/bin/pip install -r requirements-pi.txt

# 4. YAMNet 최초 다운로드
# 최초 한 번은 인터넷 연결이 필요하다.
TFHUB_CACHE_DIR="$HOME/.cache/tfhub" \
./venv/bin/python -c \
  "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet OK')"

# 5. ReSpeaker 장치 확인
./venv/bin/python main.py --list-devices

# 6. BLE 없이 마이크 추론 확인
./venv/bin/python main.py --no-ble

# 7. 마이크 추론 + Bluetooth 실행
./venv/bin/python main.py
```

## 시스템 구조

```text
ReSpeaker Mic Array v3.0
  → 16 kHz mono, channel 0
  → 2초 sliding window (기본 hop 1초)
  → 세 전문 검출기의 독립 sigmoid 추론
  → 클래스별 threshold 판정
  → 기존 BLE GATT JSON notify
```

동일 클래스를 여러 모델이 투표하는 앙상블이 아니다. 각 검출기가 서로 다른
클래스 그룹을 담당한다. 다른 검출기의 raw 확률을 직접 비교하지 않는다.

| 검출기 | 담당 클래스 |
|---|---|
| YAMNet + 학습 head | `siren`, `scream`, `crying` |
| EfficientAT + 학습 head | `bicycle_bell`, `baby_cry`, `gunshot`, `glass_break` |
| MobileNetV4-Small | `water`, `knock`, `dog`, `cat` |

`경보`와 `화재경보`는 `siren` 하나로 통합한다. 전체 클래스는 11개다.

## 저장소 업데이트

Pi에서 최신 코드를 받을 때:

```bash
cd ~/EdgeAudioRecognition
git pull origin ensemble-inference
cd jetson_backend
```

requirements가 변경됐다면 다시 설치한다.

```bash
./venv/bin/pip install -r requirements-pi.txt
```

## 설치 확인

모델 라이브러리:

```bash
./venv/bin/python -c \
  "import torch, torchaudio, timm; print(torch.__version__, torchaudio.__version__, timm.__version__)"

./venv/bin/python -c \
  "import tensorflow as tf, tensorflow_hub as hub; print(tf.__version__)"
```

설치된 버전 저장:

```bash
./venv/bin/pip freeze > requirements-pi.lock.txt
```

## IMU (MPU9250)

기기가 흔들리는 동안에는 손이나 옷에 쓸리는 마찰음이 마이크에 그대로 들어와
오탐이 난다. IMU를 켜면 오디오 윈도우와 같은 구간의 움직임 상태를 함께 실어
보내고, 필요하면 그 구간의 추론을 아예 건너뛸 수 있다. `--imu`를 주지 않으면
IMU 코드는 로드되지 않으므로 기존 동작 그대로다.

### 1. I2C 활성화와 연결 확인

```bash
sudo raspi-config   # Interface Options -> I2C -> Enable
sudo apt install -y i2c-tools
./venv/bin/pip install smbus2

./venv/bin/python test_imu.py --scan
```

`0x68 <- IMU 후보 MPU9250`처럼 나오면 정상이다. 주소가 `0x69`면
`--imu-address 0x69`를 쓴다.

### 2. 캘리브레이션

정지 상태에서 합성 가속도가 1g가 되도록 스케일을 맞추고, 자이로 바이어스를
측정한다. 기기를 움직이지 않는 곳에 두고 한 번만 실행하면 되고, 결과는
`imu_calibration.json`에 저장된다. 기울어져 있어도 상관없다.

```bash
./venv/bin/python test_imu.py --calibrate
```

`검증(정지 상태 1초): motion=still`이 나오면 성공이다.

### 3. 단독 동작 확인

```bash
./venv/bin/python test_imu.py --window-seconds 1.0
```

가만히 두면 `still`, 손으로 들면 `motion`, 툭 치면 `shock`, 떨어뜨리면
`free_fall`이 나온다.

### 4. 추론과 함께 실행

```bash
# 움직임 상태를 결과에 붙이기만 한다
./venv/bin/python main.py --imu

# 움직이는 동안에는 추론을 건너뛴다 (오탐 억제 + CPU 절약)
./venv/bin/python main.py --imu --suppress-on-motion
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--imu` | 꺼짐 | IMU 샘플링을 켠다 |
| `--imu-bus` | `1` | I2C 버스 번호 |
| `--imu-address` | `0x68` | I2C 주소 |
| `--imu-sample-hz` | `50` | 샘플링 주기 |
| `--suppress-on-motion` | 꺼짐 | `motion` 구간의 추론을 건너뛴다 |

`--suppress-on-motion`은 `motion`만 막는다. `shock`과 `free_fall`은 유리가
깨지거나 물건이 떨어지는 순간이라 오히려 소리를 놓치면 안 되므로 추론을 그대로
돌리고 `motion` 필드로 표시만 한다.

### 움직임 상태

| 상태 | 의미 | 판정 기준 |
| --- | --- | --- |
| `still` | 정지 | 가속도 RMS ≤ 0.04g 이고 자이로 ≤ 12 dps |
| `motion` | 들고 있거나 걷는 중 | 위 조건을 벗어남 |
| `shock` | 충격 | 가속도 피크 ≥ 2.2g 또는 자이로 ≥ 400 dps |
| `free_fall` | 낙하 | 합성 가속도 < 0.35g 가 0.08초 이상 지속 |
| `unknown` | IMU 없음/통신 끊김 | 샘플 부족 |

임계값은 `motion_state.py` 상단 상수로 모여 있다.

IMU 읽기가 연속 10회 실패하면 상태가 `unknown`으로 떨어지고 오디오 파이프라인은
그대로 돌아간다. IMU 문제가 소리 감지를 멈추지는 않는다.

## YAMNet과 인터넷 연결

YAMNet 백본은 최초 한 번 TF-Hub에서 다운로드한다.

```bash
TFHUB_CACHE_DIR="$HOME/.cache/tfhub" \
./venv/bin/python -c \
  "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet cache OK')"
```

다운로드 후에는 `$HOME/.cache/tfhub` 캐시를 사용하므로 오프라인으로 다시 실행할
수 있다.

### Pi에서 인터넷을 사용할 수 없는 경우

인터넷이 되는 다른 머신에서 캐시를 만든다.

```bash
python3 -m venv yamnet-cache-venv
./yamnet-cache-venv/bin/pip install tensorflow tensorflow_hub "setuptools<81"

mkdir -p tfhub_cache
TFHUB_CACHE_DIR="$PWD/tfhub_cache" \
./yamnet-cache-venv/bin/python -c \
  "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('cached')"

tar czf yamnet_tfhub_cache.tar.gz tfhub_cache
```

`yamnet_tfhub_cache.tar.gz`를 Pi로 복사한 뒤:

```bash
mkdir -p "$HOME/edge-audio-cache"
tar xzf yamnet_tfhub_cache.tar.gz -C "$HOME/edge-audio-cache"

cd ~/EdgeAudioRecognition/jetson_backend

TFHUB_CACHE_DIR="$HOME/edge-audio-cache/tfhub_cache" \
./venv/bin/python main.py --no-ble
```

자동 실행 환경에서도 같은 `TFHUB_CACHE_DIR`을 지정해야 한다.

## 권장 점검 순서

문제를 모델, 마이크, BLE 단계로 분리하기 위해 다음 순서대로 확인한다.

### 1. ReSpeaker 장치 확인

```bash
./venv/bin/python main.py --list-devices
```

자동 탐색이 실패하면 출력된 장치 index를 지정한다.

```bash
./venv/bin/python main.py \
  --device-index 2 \
  --channel-index 0 \
  --no-ble
```

ReSpeaker Mic Array v3.0의 6채널 펌웨어에서는 기본적으로 channel 0을 사용한다.

### 2. WAV 파일로 모델 확인

16 kHz mono WAV가 가장 좋다. 다른 sample rate와 stereo 파일도 실행 시 변환한다.

```bash
./venv/bin/python main.py \
  --no-ble \
  --input-wav /path/to/test.wav
```

특정 검출기만 확인:

```bash
./venv/bin/python main.py \
  --no-ble \
  --input-wav /path/to/test.wav \
  --detectors yamnet
```

### 3. Pi 처리 속도 측정

```bash
./venv/bin/python benchmark.py \
  --iters 30 \
  --warmup 3 \
  --hop-seconds 1.0
```

- `REALTIME OK`, exit code 0: p95 추론 시간이 hop 예산 이내
- `NEEDS ATTENTION`, exit code 2: 기능 오류가 아니라 처리 속도 부족

속도가 부족하면 다음 순서로 확인한다.

```bash
# 1. 50% overlap 제거
./venv/bin/python main.py --hop-seconds 2.0 --no-ble

# 2. 검출기 병렬 실행 측정
./venv/bin/python benchmark.py --concurrent

# 3. 개별 검출기 병목 확인
./venv/bin/python benchmark.py --detectors yamnet
```

### 4. 실시간 마이크 추론

```bash
./venv/bin/python main.py --no-ble
```

### 5. 실시간 마이크 추론과 BLE

```bash
./venv/bin/python main.py
```

## 자주 쓰는 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--hop-seconds` | `1.0` | 2초 윈도우 사이 간격 |
| `--detectors` | 3개 전체 | 실행할 전문 검출기 목록 |
| `--min-db` | `45` | `-45 dBFS` 미만을 저음량으로 처리 |
| `--no-skip-low-db` | 꺼짐 | 저음량도 모델에 전달 |
| `--debounce-seconds` | `3.0` | 같은 클래스 반복 감지 구간 |
| `--concurrent` | 꺼짐 | 검출기 스레드 병렬 실행 |
| `--torch-threads` | 최대 `4` | PyTorch CPU thread 수 |
| `--no-ble` | 꺼짐 | BLE 없이 콘솔만 실행 |
| `--debug` | 꺼짐 | 전체 확률, threshold, 모델별 지연 출력 |

체크포인트나 모델을 다른 위치에 둘 때:

```bash
./venv/bin/python main.py \
  --checkpoints-dir /opt/edge-models/checkpoints \
  --thresholds /opt/edge-models/thresholds.json \
  --efficientat-dir /opt/edge-models/EfficientAT
```

환경변수도 사용할 수 있다.

```text
EDGEAUDIO_CHECKPOINTS_DIR
EDGEAUDIO_THRESHOLDS
EDGEAUDIO_EFFICIENTAT_DIR
```

## 클래스별 threshold

기본 파일:

```text
edge_audio_pi/parallel_detectors/report/thresholds.json
```

각 클래스는 자신의 threshold만 사용한다.

```json
{
  "thresholds": {
    "siren": 0.46013665,
    "bicycle_bell": 0.32903698,
    "knock": 0.9999893
  }
}
```

예를 들어 `knock=0.95`와 `bicycle_bell=0.40`을 raw 확률만으로 비교하지 않는다.
`knock`은 자신의 threshold를 넘지 못했고 `bicycle_bell`은 threshold를 넘었으므로
자전거 벨만 후보가 된다.

대표 `label`은 후보 중 `probability - threshold`가 가장 큰 클래스다. threshold를
임의로 낮추면 false positive가 증가하므로, 검증 데이터에서 다시 튜닝한 값이 없다면
원본 파일을 유지한다.

## BLE 계약과 JSON

기존 BLE 코드를 그대로 사용한다.

```text
Device name        JHello
Service UUID       12345678-1234-5678-1234-56789abcdef0
Characteristic     12345678-1234-5678-1234-56789abcdef1
Frame              #<seq>:<part>/<total>:<JSON 조각>
```

기존 JSON 필드는 삭제하거나 이름을 바꾸지 않았다. 새 필드는 `candidates`와
`repeat` 두 개다.

```json
{
  "source": "live_inference_refactored_ble_independent",
  "time": "12:34:56",
  "label": "siren",
  "score": 0.91,
  "status": "detected",
  "status_text": "감지",
  "level_dbfs": -21.4,
  "enhanced_dbfs": null,
  "quiet_gain": null,
  "loud_gain": null,
  "clipped": false,
  "scores": {
    "siren": 0.91,
    "scream": 0.03,
    "glass_break": 0.72
  },
  "raw": "[12:34:56] 감지: siren ...",
  "candidates": ["siren", "glass_break"],
  "repeat": false
}
```

- `label`: threshold margin이 가장 큰 대표 감지
- `score`: 대표 감지의 sigmoid 확률
- `candidates`: 각자 threshold를 넘은 모든 클래스
- `repeat`: debounce 시간 안에 같은 후보가 다시 감지됐는지 여부
- `scores`: 활성화된 검출기의 전체 클래스 확률
- `motion`: 같은 구간의 IMU 요약. IMU를 켜지 않으면 `state`가 `unknown`이다

```json
"motion": {
  "state": "still",
  "samples": 100,
  "accel_rms_g": 0.0028,
  "accel_peak_g": 1.0089,
  "gyro_max_dps": 0.56,
  "pitch_deg": 34.6,
  "roll_deg": 73.8,
  "temperature_c": 29.0
}
```

`--suppress-on-motion`으로 건너뛴 윈도우는 `status`가 `motion_skipped`,
`label`이 움직임 상태(`motion`/`shock`/`free_fall`)로 나간다. 앱에서 `shock`과
`free_fall`은 소리와 별개로 알림에 쓸 수 있다.

기존 앱은 `label`만 읽어도 계속 동작한다.

## 문제 해결

### `No module named sounddevice` 또는 PortAudio 오류

```bash
sudo apt install -y portaudio19-dev
./venv/bin/pip install sounddevice
```

### ReSpeaker가 검색되지 않음

```bash
arecord -l
./venv/bin/python main.py --list-devices
```

장치는 보이지만 자동 탐색이 안 되면 `--device-index`를 지정한다.

### YAMNet 다운로드 실패

인터넷 연결과 시스템 시간을 확인한다.

```bash
date
curl -I https://tfhub.dev/google/yamnet/1
```

인터넷을 사용할 수 없다면 위의 TF-Hub 캐시 사전 준비 절차를 사용한다.

### TensorFlow 또는 PyTorch 설치 실패

```bash
uname -m
python3 --version
```

기준 환경은 Raspberry Pi OS 64-bit Bookworm, `aarch64`, Python 3.11이다.
32-bit OS나 너무 새로운 Python 버전에서는 필요한 wheel이 없을 수 있다.

### BLE만 실패

WAV 추론과 `--no-ble` 실행은 성공하지만 BLE 실행만 실패한다면 모델 문제가 아니다.

```bash
sudo systemctl status bluetooth
sudo systemctl restart bluetooth
bluetoothctl show
```

venv를 `--system-site-packages`로 만들었는지 확인한다.

### 추론 결과가 과거로 밀림

입력 큐가 포화되면 백엔드는 오래된 오디오를 버리고 폐기 누계를 출력한다.
`benchmark.py`로 병목을 측정하고 `--hop-seconds 2.0` 또는 `--concurrent`를 검토한다.

## 검증 상태

개발 머신에서 다음 항목을 확인했다.

- bounded audio queue와 drop-oldest
- 2초 sliding window와 hop
- 클래스별 threshold와 margin 순위
- 병렬 검출기 확률 병합
- 기존 BLE JSON 필드와 `candidates`/`repeat`
- 실제 체크포인트 3개 로딩과 11클래스 추론

개발 머신의 테스트 결과가 Raspberry Pi 5의 실제 처리 속도, ReSpeaker 장치 이름,
BlueZ 권한까지 보장하지는 않는다. Pi에서는 반드시 이 문서의 점검 순서와
`benchmark.py`를 실행한다.
