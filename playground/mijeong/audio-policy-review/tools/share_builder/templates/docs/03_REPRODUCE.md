# 실험 재현 순서

저장소 루트에서 실행한다. 먼저 `git lfs pull`, 환경 설치, `python scripts/verify_share.py`를 완료한다.

## A. WAV 없이 기존 정책 비교 재현

```powershell
conda activate audio-policy
python -B -m unittest discover -s src/policy_v1_4 -p "test_*.py" -v
python scripts/replay_cached.py --stage baseline --timing audio_clock
python scripts/replay_cached.py --stage baseline --timing independent_worker_sim
python scripts/replay_cached.py --stage adaptive --timing independent_worker_sim
python scripts/replay_cached.py --stage temporal --timing independent_worker_sim
python scripts/replay_cached.py --stage wait_all --timing independent_worker_sim
python scripts/replay_cached.py --stage bounded --timing independent_worker_sim
```

각 명령은 `.local/replays/실행시각`의 새 폴더에서 실행한다. 원본 결과를 덮어쓰지 않는다. `--out .local/replays/고유이름`으로 지정할 수도 있다. 이 명령들은 저장된 3모델 예측으로 정책 결과를 재계산하며 원음 추론 시간이나 GPU 지연을 새로 측정하지 않는다.

공용 입력은 기본적으로 `results/original/V14_pilot_20261016_034601`에서 읽는다. followup에 중복 입력을 복사하지 않는다. 공유본에서 생략한 trace도 위 명령으로 다시 생성된다.

## B. 저장된 validation 점수로 정적 비교

```powershell
conda activate audio-policy
$repo = (Get-Location).Path
$project = Join-Path $repo 'project'
$policy = Join-Path $repo 'src\policy_v1_4'
$out = Join-Path $repo ('.local\runs\V14_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
python "$policy\run_framework_pilot.py" --project-root "$project" --assignment predefined --ratio 0.70 --seed 42 --fusions F0 F1 F2 --owner-background global_max --target-precision 0.95 --minimum-accepted 20 --recall-floor 0.5 --out "$out"
```

이후 단계에서도 같은 출력 경로를 사용한다.

## C. 팀원 PC의 원음으로 새 window 추론

앞 단계와 같은 PowerShell 터미널·`$out`을 유지한다. 아래 dataset 경로는 팀원 PC에 맞춘다.

```powershell
$dataset = 'D:\Sound_Dataset_aug'
python "$policy\prepare_replay.py" --selection --out "$out" --max-per-class 8 --phase-offsets 0 0.5 1 1.5 --seed 42 --old-dataset-root /home/plass/YAMNet/aug_data --dataset-root "$dataset"
python "$policy\check_replay_audio.py" --out "$out"
```

audio_preflight.json의 passed=true, failed_files=0을 확인한다. preflight에는 soundfile이 필요하므로 PyTorch 공통 의존성까지 설치된 audio-policy에서 실행한다.

```powershell
python "$policy\export_replay_predictions.py" --selection --project-root "$project" --out "$out" --model efficientat --device cuda
python "$policy\export_replay_predictions.py" --selection --project-root "$project" --out "$out" --model mobilenetv4 --device cuda
conda activate audio-yamnet
python "$policy\export_replay_predictions.py" --selection --project-root "$project" --out "$out" --model yamnet --device cpu
```

세 replay_*.npz가 모두 생성되면 평가한다.

```powershell
conda activate audio-policy
python "$policy\evaluate_framework_replay.py" --out "$out" --fusions F0 F1 F2 --rules R0 R1 R2 --inputs I0 I1 I2 --temporal T0 --timing audio_clock --async-policy wait_all --result-tag B_expanded
python "$policy\evaluate_framework_replay.py" --out "$out" --fusions F0 F1 F2 --rules R0 R1 R2 --inputs I0 I1 I2 --temporal T0 --timing independent_worker_sim --async-policy wait_all --result-tag B_expanded_fifo
```

manifest/config를 변경하면 preflight와 세 모델 export를 다시 실행한다. 기존 결과 JSON의 경로를 단순 치환하면 fingerprint가 바뀌므로 새 실험 폴더에서 설정을 생성한다.

## 결과 공유

개인 결과는 `.local`에 남는다. 공유할 실행이 정해지면 결과 폴더를 `results/team_runs/본인이름/실험ID`로 복사하고 실행 조건·환경·데이터 group 목록을 함께 기록한다. 이어서 `python scripts/update_inventory.py`, `python scripts/verify_share.py`를 실행한 뒤 Git에 추가한다. 기존 캡처 결과는 비교 기준으로 보존한다.
