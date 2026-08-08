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

## IMU 흔들림 보정 (MPU-9250)

키링이 흔들리면 DSP가 주는 DOA도 같이 흔들린다. 소리는 가만히 있는데 화면의
방향만 출렁인다. `imu.py`가 기기가 **자기 기준 방위에서 얼마나 돌아갔는지**(`swing`)를
추적해서 그만큼 되돌려 준다.

```text
angle = (raw_DOA + sign*swing - north_offset) % 360
```

절대 방위(북쪽)는 쓰지 않는다. `yaw`와 `ref`는 같이 드리프트하므로 그 차이인
`swing`에서는 드리프트가 상쇄된다. 지자기(AK8963)는 초기화도 하지 않는다.
I2C bus 1 / `0x68`에 물려 있고 배터리 게이지(`0x36`)와 버스를 공유한다.
`smbus2`로 레지스터를 직접 읽으므로 새 의존성이 없다.

### 흔들림과 방향 전환 구분

둘 다 0.5~2 Hz 대역이라 lowpass로는 갈라지지 않는다. **평균만 보는 판별식도
쓸 수 없다.** 진동을 자기 주기보다 짧은 창으로 평균내면 상쇄가 아니라 정류가
되기 때문이다. 창 `T`에서 진폭 `A`, 주파수 `f` 진동의 최악 평균은
`2A·|sin(πfT)|/T`이고, ±25도 흔들림을 0.5초로 평균내면

| 흔들림 | 0.5초 평균 |
|---|---:|
| 0.5 Hz | 71 deg/s |
| 0.75 Hz | 92 deg/s |
| 1.0 Hz | 100 deg/s |

이 나온다. 실제 90도 회전(1.5초)은 60 deg/s뿐이라 **흔들림이 회전보다 큰 값**이
된다. 창을 늘려도 4초에서는 같은 회전이 22.5 deg/s로 떨어져 반대로 놓친다.

그래서 크기 조건에 **회전 방향 반전 횟수**를 함께 본다.

```text
|mean(yaw_rate, W)| > turn_threshold
  그리고 창 W 안에서 회전 방향이 한 번도 뒤집히지 않음  -> 방향 전환
그 외                                                  -> 흔들림
```

반전 횟수는 진폭·위상에 관계없이 `f` Hz 진동이면 초당 `2f`번이므로 aliasing이
없다. 기본 `W=1.2초`는 `1/(2W)=0.42 Hz` 위의 모든 흔들림을 잡아낸다.

### 부호 측정

`--imu-swing-sign`은 **반드시 실측해야 한다.** 펌웨어가 DOA 방향을
"Orientation depends on build configuration"이라고만 밝히기 때문에 조회할 방법이
없다. 부호가 반대면 보정이 흔들림을 빼는 대신 더해서 **흔들림이 2배가 된다.**

연속적인 소리(음악·물소리·선풍기)를 켜 두고, 소리와 몸은 고정한 채 키링만
좌우로 ±60도쯤 계속 돌리면서:

```bash
sudo ./venv/bin/python imu_sign_check.py --seconds 25
```

`d(DOA)/d(yaw)` 기울기를 회귀로 구해 `--imu-swing-sign` 값을 출력한다.
`R^2`가 낮으면 소리가 끊겼거나 회전이 부족한 것이니 다시 실행한다.

### 확인

```bash
# IMU 단독 - 정지 드리프트와 회전 반응
./venv/bin/python -c "
from imu import IMUReader
import time
r = IMUReader()
print('describe:', r.describe())
for _ in range(30):
    time.sleep(0.5); print(r.debug_snapshot())
"

# 전체 실행
sudo ./venv/bin/python main.py --north-offset 90 --ble-chunk-bytes 500
```

시작 시 자이로 바이어스를 1.5초 평균으로 잡는다. **이 동안 정지해 있어야 한다.**
움직이면 표준편차 경고를 내고 그대로 진행하므로, 경고가 보이면 재시작한다.

보드가 수평이 아니면 자이로 Z는 yaw가 아니다. 기본값 `--imu-yaw-axis gravity`는
가속도로 측정한 중력축에 자이로 벡터를 투영하므로 어느 각도로 달아도 동작하고,
보드가 수평일 때는 자이로 Z와 정확히 같아진다.

BLE 패킷에는 `imu_status` 문자열 하나만 추가된다. `swing`/`yaw`/`ref` 같은
디버그 값은 509바이트 상한 때문에 콘솔(stderr)로만 나간다.

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
| `--disable-imu` | 꺼짐 | IMU 흔들림 보정을 끄고 DOA 원값을 그대로 사용 |
| `--imu-swing-sign` | `+1` | `imu_sign_check.py`로 측정한 부호 (`+1` 또는 `-1`) |
| `--imu-turn-threshold` | `10.0` | 방향 전환으로 볼 yaw 각속도(deg/s) 하한 |
| `--imu-turn-window` | `1.2` | 방향 전환 판정 창(초). 흔들림 반주기보다 길어야 한다 |
| `--imu-ref-tau-still` | `4.0` | 정지/흔들림 상태에서 기준 방위 시정수(초) |
| `--imu-ref-tau-turn` | `0.25` | 방향 전환 중 기준 방위 시정수(초) |
| `--imu-yaw-axis` | `gravity` | `gravity`=중력축 투영, `z`=자이로 Z 그대로 |

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
