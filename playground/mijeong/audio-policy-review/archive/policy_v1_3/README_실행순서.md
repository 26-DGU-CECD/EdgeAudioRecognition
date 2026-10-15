# 무투표 selection replay v1.3 실행 순서

기준 데이터: `D:\Sound_Dataset_aug`

기존 14-class specialist의 S0/S1/S2를 원음 replay와 연결하는 별도 실행 묶음입니다.
담당 라벨의 점수 중 최댓값을 선택하며, 가중 voting이나 모은 점수의 재정규화는 하지 않습니다.
checkpoint·전처리·담당 배정·S01 임계값을 유지하고 입력 갱신 간격 H만 2/1/0.5초로 비교합니다.
임계값은 다시 fitting하지 않습니다. 실제 최고 성능 무투표 run으로 확인된 구성이라는 주장은 하지 않습니다.

## 현재 준비된 상태

- S01 seed 42의 설정을 동결했습니다. seed 42는 S01과 연결하기 위한 기준이며 성능이 좋아서 선택한 seed가 아닙니다.
- 같은 평가 group에서 라벨별 원음 1개씩, 총 14개를 선정했습니다. 보정 group과 교차하지 않습니다.
- Linux path 식별자를 보존하고 실제 읽기 경로만 D:로 연결했습니다.
- 원음·checkpoint·원본 프로젝트 주요 코드·이번 replay 코드의 hash를 기록했습니다.
- 별도 결과 폴더 `C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\selection-replay-S03_smoke`를 사용합니다.
- 새 모델 학습이나 실제 모델 추론은 아직 수행하지 않았습니다.

## 1. VS Code에서 열 폴더

```text
C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\selection-replay-v1_3
```

아래 명령은 Windows PowerShell 기준입니다. 같은 터미널에서 환경을 바꾸면 변수는 유지됩니다.
새 터미널을 열었다면 변수를 다시 지정하세요.

```powershell
$replayCode = "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\selection-replay-v1_3"
$audioProject = "C:\Users\PLASS\Desktop\audio_experiments_DEF"
$replayOut = "C:\Users\PLASS\Documents\Codex\2026-10-16\r\outputs\selection-replay-S03_smoke"
Set-Location $replayCode
```

## 2. 준비된 원음과 코드 확인

```powershell
conda activate audio-policy
python -m unittest discover -p "test_*.py" -v
python probe_selection_audio.py --out "$replayOut"
```

probe는 NumPy와 Python 표준 라이브러리로 선택한 14개 WAV의 헤더와 최초 2초를 읽습니다.
모델 전처리나 모델 추론을 실행하는 단계는 아닙니다. `selection_audio_probe.json`을 확인하세요.

## 3. Torch 환경 설치 및 GPU 확인

현재 확인 시점에는 `audio-policy`만 설치되어 있으며 이 환경에는 Torch, TensorFlow, SciPy, SoundFile이 없습니다.
`audio-torch`가 이미 있으면 create 명령은 건너뛰고 activate부터 실행하세요.

```powershell
conda create -n audio-torch python=3.11 pip -y
conda activate audio-torch
python -m pip install -r .\requirements\requirements-torch-cuda128.txt
python -m pip install -r .\requirements\requirements-torch.txt
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda'))"
```

`requirements` 폴더에 앞서 정리한 설치 파일을 함께 담았습니다.
원본 모델 생성자는 pretrained 모델 초기화도 호출하므로 첫 실행에 기존 캐시 또는 네트워크가 필요할 수 있습니다.
실험 가중치는 생성 후 준비된 specialist checkpoint에서 `strict=True`로 로드합니다.

## 4. EfficientAT, MobileNetV4 실제 원음 추론

```powershell
python export_selection_replay.py --project-root "$audioProject" --out "$replayOut" --model efficientat --device cuda
python export_selection_replay.py --project-root "$audioProject" --out "$replayOut" --model mobilenetv4 --device cuda
```

한 모델을 로드해 유지한 채 batch=1로 실행합니다. 각 모델은 14개 episode × 9개 window = 126개 결과를 만듭니다.
초기 warm-up 2회는 별도입니다. 모델별 결과를 덮어쓰지 않으며, 이미 결과가 있으면 중단합니다.
재실행이 필요하면 새 output 폴더를 prepare하거나 기존 결과를 명시적으로 보존·이동한 뒤 진행하세요.

## 5. YAMNet 환경과 실제 원음 추론

Windows native의 TensorFlow 2.20은 CPU로 기능을 점검합니다.
`audio-yamnet`이 이미 있으면 create를 건너뛰세요.

```powershell
conda create -n audio-yamnet python=3.11 pip -y
conda activate audio-yamnet
python -m pip install -r .\requirements\requirements-yamnet.txt
python -m pip check
python -c "import tensorflow as tf,tensorflow_hub as hub,tf_keras; print(tf.__version__,hub.__version__,tf_keras.__version__)"
python export_selection_replay.py --project-root "$audioProject" --out "$replayOut" --model yamnet --device cpu
```

YAMNet은 원음에서 embedding을 새로 계산하고 mean pooling한 뒤 기존 head를 실행합니다.
저장 embedding에서 head만 실행하는 것이 아닙니다. 최초 TF Hub 로딩에 기존 캐시 또는 네트워크가 필요합니다.
세 모델 모두 GPU인 처리시간 실험은 향후 WSL2 또는 실제 배포 환경에서 별도로 진행합니다.
현재 manifest는 Windows 경로를 사용하므로 그대로 WSL에서 실행하는 구성은 아닙니다.

## 6. 기존 정적 예측과 연결이 맞는지 확인

```powershell
conda activate audio-policy
python check_selection_reference.py --out "$replayOut"
```

각 episode의 `[2,4]초` window는 원래 2초 clip 전체입니다.
모델별 이 window의 점수를 원래 `val_predictions.npz`의 같은 path와 대조합니다.
기본 점검 기준은 softmax 점수 최대 차이 0.001 이하 및 top-1 일치입니다.
이는 진단 기준이며 수치 호환성·정답률을 보장하는 기준이 아닙니다.
실패하면 `selection_reference_check.json`의 라벨·모델별 차이를 먼저 확인하세요.
단순히 허용 오차를 크게 바꿔 통과시키지 말고 checkpoint, 전처리, 프레임워크 차이를 확인합니다.

## 7. 입력 간격과 출력 정책 비교

```powershell
python evaluate_selection_replay.py --out "$replayOut" --background-mode all
```

3개 background 가정 × 3개 hop × S0/S1/S2 = 27개 비교 행이 생성됩니다.
같은 0.5초 grid cache에서 1초/2초 위치를 추출하므로 모델을 27회 다시 실행하지 않습니다.

| 파일 | 확인할 내용 |
|---|---|
| selection_audio_probe.json | 원음 14개 읽기와 checkpoint hash 점검 |
| selection_replay_각모델.npz | 실제 새 원음 추론 logits 및 서비스 시간 |
| selection_runtime_각모델.json | 모델별 환경·버전·소요시간 |
| selection_reference_check.json | 원래 2초 clip 예측과의 호환성 |
| selection_replay_comparison.csv | 27개 정책 조합 |
| selection_replay_per_episode.csv | 라벨별 첫 정답·잘못된 알림·미검출 |
| selection_replay_window_decisions.csv | window별 최고 후보와 accepted/background/unconfirmed 상태 |
| selection_replay_scope.json | 평가 해석 범위 |

## 해석 범위

- 한 episode는 2초 zero silence + 원음 선두 2초 + 2초 zero silence입니다.
- 삽입 시점은 실제 사건 onset이 아닙니다. 실제 2초 내 표시 목표 달성으로 해석하지 않습니다.
- 삽입 위치가 실행 grid와 정렬된 초기 연결 점검입니다. 후속에는 시작 위상 변화·실제 연속 배경을 추가합니다.
- 14개 원음은 코드 연결 점검용이며 정책 순위를 확정하는 표본이 아닙니다.
- S01의 S2 `fire_alarm`/`knock` null도 그대로 유지합니다. replay 연결을 위해 임계값을 임의 변경하지 않았습니다.
- background 오알림은 원음과 겹치는 구간뿐 아니라 episode 전체에서 셉니다.
- 서비스 시간은 모델별 독립 실행이며, 세 모델 동시 실행·큐·IPC·UI·실기 지연은 포함하지 않습니다.
- CPU YAMNet과 GPU Torch의 비용을 합친 값을 세 모델 GPU 처리시간으로 표현하지 않습니다.
- 무음에서도 모델을 실행합니다. 전처리 hard gate나 시간 연속 확정·해제는 새로 도입하지 않았습니다.

## 다음 규모로 늘릴 때

처음 14개 연결 점검이 통과한 뒤 새 결과 폴더에서 라벨당 최대 8개로 준비합니다.
최대 수는 실제 접근 가능한 평가 group 수에 따라 달라지므로 manifest의 class_counts를 확인하세요.

```powershell
conda activate audio-policy
$replayOut = Join-Path $audioProject ("policy_results\S03_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
python prepare_selection_replay.py --project-root "$audioProject" --selection-config "$audioProject\policy_results\S01_20261016_022757\selection_config.json" --dataset-root "D:\Sound_Dataset_aug" --max-per-class 8 --out "$replayOut"
```

새로 준비한 폴더에 대해 세 모델 export, reference check, evaluate를 모두 다시 실행합니다.
manifest·코드·원음을 바꾼 이전 cache와 섞지 않습니다.

## 출처

원본 v1.2의 policy_core.py, selection_core.py, project_io.py와 핵심 테스트는 그대로 복사했습니다.
export_selection_replay.py는 원본 export_replay_predictions.py의 resident worker와 전처리를 재사용하고,
설정·checkpoint 해석·cache 출처 검사를 selection 전용으로 연결했습니다.
원본 프로젝트와 원본 policy-pilot은 수정하지 않았습니다.
