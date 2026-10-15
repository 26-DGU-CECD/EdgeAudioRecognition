# Audio policy experiments — team snapshot

Windows / VS Code / Conda 기반 오디오 specialist와 판단 정책 실험의 공유용 스냅샷이다.

처음 읽을 문서:

1. [경로와 폴더 구성](docs/01_PATHS_AND_MODELS.md)
2. [환경 설치](docs/02_ENVIRONMENT.md)
3. [결과 재현과 원음 재추론](docs/03_REPRODUCE.md)
4. [GitHub 첫 push·팀원 clone·이후 pull/push](docs/04_GITHUB_WORKFLOW.md)
5. [현재 결과와 한계](docs/05_RESULTS.md)

## 포함 범위

- `project/`: 기존 audio_experiments_DEF의 학습·평가 코드, metadata, specialist_metadata, runs, EfficientAT vendor 소스 및 LICENSE. runs의 학습 checkpoint를 모두 포함한다.
- `src/policy_v1_4/`: 현재 판단 정책 코드, 테스트 42개, 설계 문서, 제공된 참고 결과.
- `archive/`: 초기 검토 코드/문서, v1/v1.1/v1.2 ZIP, 별도로 구현했던 v1.3. 구버전을 현재 v1.4와 혼용하지 않는다.
- `environments/`: 실제 사용한 requirements와 설치된 환경의 패키지 버전 스냅샷.
- `results/original/`: S01, S02, V14 smoke/expanded 등 원래 실행 결과.
- `results/followup/`: 적응형 간격·시간 확인·bounded 추가 비교. 공용 예측 cache는 `results/original/V14_pilot_20261016_034601/`에 한 번만 보관한다.
- `reports/`: 결과 검토, 상세 분석, 원음 오류 목록, 이전 실행 안내.
- `manifests/`: 파일 SHA256, 원본 경로 매핑, 전체 모델 목록, 포함/제외 파일 목록.
- `scripts/`: 공유 파일 검증, 예측 cache 재현, 파일 목록 갱신.
- `.local/`: 개인 실행 결과와 로컬 파일용. Git에서 제외한다.

WAV 데이터셋 원음은 포함하지 않는다. 실제 데이터셋 경로는 원 작성자 PC의 `D:\Sound_Dataset_aug`이며 팀원은 자신의 데이터셋 경로를 지정해야 한다. 재생성 가능한 `embedding_cache`는 기본 수집에서 제외하고 목록을 남긴다. `--include-embedding-cache`로 복사해도 Git에서는 제외된다. 모델과 `.npz`는 Git LFS로 공유한다.

공유용 results는 중복 입력과 재생성 가능한 trace를 줄인 구성이다. 비교표·음원별 결과·scope는 유지한다. 제거 목록과 trace 복원 방법은 [results 안내](results/README.md)를 본다.

## 가장 빠른 결과 재현

Git LFS 파일을 받은 후 저장소 루트에서 실행한다. WAV와 GPU 없이 저장 예측으로 정책 비교를 재현할 수 있다.

```powershell
git lfs install --local
git lfs pull
conda create -n audio-policy python=3.11 pip -y
conda activate audio-policy
python -m pip install -r environments/requirements-policy.txt
python scripts/verify_share.py
python -B -m unittest discover -s src/policy_v1_4 -p "test_*.py" -v
python scripts/replay_cached.py --stage adaptive --timing independent_worker_sim
```

새 결과는 `.local/replays/실행시각`에 생성된다. 저장된 config/manifest는 내용과 줄바꿈까지 fingerprint에 포함되므로, 기존 결과의 절대 경로를 일괄 치환하지 않는다. 원음 재추론은 새 config/manifest를 생성해 진행한다.

최종 배포 정책이나 독립 test 성능을 확정한 저장소가 아니다. 현재 유력 후보 F1/R0/I3/T0는 FIFO 모의실험에서 요청 호출을 약 21% 줄였지만 사건 오표시가 여전히 크다. 실제 동시 장치·UI 지연은 별도 검증이 필요하다.
