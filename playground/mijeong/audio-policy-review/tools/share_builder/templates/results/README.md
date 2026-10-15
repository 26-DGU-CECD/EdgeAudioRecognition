# 공유 결과 폴더 구성

## 남겨 둔 결과

- `original/S01_*`, `original/S02_*/seed_*`: 초기 정책과 seed 비교. 실험별 설정과 표본을 유지한다.
- `original/V14_smoke_*`: 14원본 연결 점검의 비교표·음원별 결과·원음 예측 입력.
- `original/V14_pilot_20261016_034601`: 확대 실험의 정적 비교, audio_clock/FIFO 비교, 음원별 결과와 **공용 replay 입력**.
- `followup`: 적응형·시간 확인·bounded의 비교표, 음원별 결과, 실행 조건 scope. 입력은 위 확대 실험과 동일하므로 그 위치를 참조한다.
- `checks/v14_reference/verification.json`: 처음 코드 검토 때의 검증 기록.
- `codex/S03_smoke`: 이전 v1.3 단계 기록. v1.4와 다른 실험이므로 유지한다.

## 제거한 항목

1. `*_trace.csv`: 세부 window 추적 로그. 저장된 예측으로 다시 생성할 수 있어 공유 snapshot에서 제외했다. 음원별 결과는 유지한다.
2. followup에 복사돼 있던 config/manifest/예측 NPZ/runtime/preflight/표본 파일 중 확대 실험 입력과 SHA256이 같은 것.
3. 검토용 reference 및 smoke의 정적 파일 중 이미 남겨 둔 파일과 SHA256이 같은 것.

다른 seed·다른 원음 표본은 합치지 않았고, 비교표나 음원별 결과의 중복 행을 임의로 삭제하지 않았다. 파일별 제거 이유·SHA256·남겨 둔 위치는 `manifests/results_cleanup.json`에 있다. 원본 위치는 `manifests/source_inventory.csv`에 보존한다.

## 필요한 trace를 다시 생성하기

저장소 루트에서 실행한다. WAV·GPU 추론은 필요 없고 새로운 `.local/replays/실행시각` 폴더에 전체 결과와 trace가 생성된다.

```powershell
conda activate audio-policy
python scripts/replay_cached.py --stage baseline --timing audio_clock
python scripts/replay_cached.py --stage baseline --timing independent_worker_sim
python scripts/replay_cached.py --stage adaptive --timing audio_clock
python scripts/replay_cached.py --stage adaptive --timing independent_worker_sim
python scripts/replay_cached.py --stage temporal --timing audio_clock
python scripts/replay_cached.py --stage temporal --timing independent_worker_sim
python scripts/replay_cached.py --stage wait_all --timing independent_worker_sim
python scripts/replay_cached.py --stage bounded --timing independent_worker_sim
```

smoke trace는 `--source results/original/V14_smoke_20261016_032646`로 입력을 지정해 재생성할 수 있다. helper의 baseline 단계는 모든 F/R/I 조합을 출력하므로 이전 smoke의 R0 결과는 그중 R0 행을 선택한다. tag 이름은 원래 파일과 다를 수 있지만 설정이 같으면 결과 값은 같다.

## 다시 전체 파일을 수집하려면

수집 kit의 build_team_share.py에 `--keep-full-results`를 지정하면 trace와 중복 사본을 보존해 새 공유 폴더를 만들 수 있다. 기존 Desktop 실험 결과와 Codex 원본 결과는 이 정리 작업에서 변경하지 않는다.
