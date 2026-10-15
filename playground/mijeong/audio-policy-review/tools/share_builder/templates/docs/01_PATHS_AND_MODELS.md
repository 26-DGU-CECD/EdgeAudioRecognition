# 경로·모델·실험 이력

## 원본 경로와 공유 폴더의 대응

| 원본 | 공유 폴더 |
|---|---|
| Desktop/audio_experiments_DEF의 코드·metadata·runs | project/ |
| audio-policy-review/audio_policy_pilot_v1_4 | src/policy_v1_4/ |
| audio-policy-review의 초기 자료와 이전 ZIP | archive/initial_review/ |
| Documents/Codex/.../outputs/selection-replay-v1_3 | archive/policy_v1_3/ |
| audio_experiments_DEF/policy_results | results/original/ |
| outputs/v1_4_expanded_next/experiment | results/followup/ |
| outputs의 결과 정리·상세 분석 | reports/ |
| D:/Sound_Dataset_aug | 외부 데이터. 포함하지 않음 |

정확한 원본 절대 경로는 `manifests/path_map.json`, 파일별 대응은 `manifests/source_inventory.csv`에 있다. 기존 보고서·JSON·CSV에 남은 절대 경로는 당시 실행 기록이다. 팀원 PC의 현재 실행 경로로 가정하지 않는다.

공유 results에서는 확대 실험과 followup의 동일한 replay 입력을 `results/original/V14_pilot_20261016_034601`에 한 번만 보관한다. 상세 trace는 재생성할 수 있어 제외했다. 파일별 대응은 `manifests/results_cleanup.json`, 복원 방법은 `results/README.md`에 있다.

`config/paths.example.json`은 경로 기록용 예시다. 기존 스크립트가 자동으로 읽는 설정 파일은 아니며 실제 실행에는 `--project-root`, `--dataset-root`를 명시한다. `archive/analysis_scripts`에는 당시 보고서 생성에 사용한 보조 Python 코드도 보존했다. 이 코드는 당시 PC의 절대 경로를 포함하므로 portable 실행 진입점인 `scripts/replay_cached.py`와 구분한다.

## 현재 v1.4에서 사용한 checkpoint

프로젝트 기준 실험 구성은 `E_predefined_specialists/nonexpert_070`, 입력 길이 W=2초다. 실제 최고 성능의 과거 run으로 확인된 구성은 아니다.

| 모델 | checkpoint 경로 | 전문 라벨 |
|---|---|---|
| EfficientAT | project/runs/E_predefined_specialists/nonexpert_070/efficientat/best.pt | bicycle, baby_cry, gun, glass_breaking |
| YAMNet head | project/runs/E_predefined_specialists/nonexpert_070/yamnet/best.keras | siren, fire_alarm, scream, cry, car_horn |
| MobileNetV4 | project/runs/E_predefined_specialists/nonexpert_070/mobilenetv4/best.pt | water, knock, dog_bark, cat_meow |

각 모델의 출력 head는 background를 포함한 공통 14클래스다. 일부 라벨 head를 0으로 채워 voting하는 실험이 아니다. YAMNet의 `.keras`는 embedding 위의 학습 head이며, 실행 시 TF Hub의 YAMNet 본체도 필요하다. 첫 실행에 TF Hub 다운로드가 필요할 수 있다. MobileNetV4/timm 등의 pretrained 생성도 로컬 cache 또는 다운로드가 필요할 수 있다.

이 세 개 외에 `project/runs`에서 발견한 다른 비율·baseline·automatic specialist checkpoint도 함께 수집한다. 전체 목록과 SHA256은 `manifests/models.json`을 본다. EfficientAT vendor의 pretrained backbone도 별도 역할로 기록한다. 외부 vendor의 LICENSE와 코드는 보존한다. 저장소 전체에 임의의 새 라이선스를 부여하지 않는다.

## 데이터와 split

- 원본 metadata는 `/home/plass/YAMNet/aug_data/...`를 기록한다. 기존 metadata를 Windows 경로로 덮어쓰지 않는다.
- 원음 재추론 시 `prepare_replay.py --old-dataset-root /home/plass/YAMNet/aug_data --dataset-root 팀원경로`로 매핑한다.
- seed 42 정적 분할: fusion-fit 5,892 clip/658 group, policy-fit 6,073/693, evaluation 11,882/1,338. group 중복은 0이다.
- policy-fit에 knock이 없다. 원음 replay 표본 확대만으로 이 보정 공백은 해결되지 않는다.
- 확대 replay는 원본 group 112개, 448 episode다. 사건 104 group×4위상=416, 배경 8×4=32다.

기존 학습 스크립트의 `project/config.py`에는 Linux 경로가 남아 있다. 현재 정책 비교·기존 checkpoint export는 `--project-root`로 재현한다. 새 학습을 하려면 별도 개발 사본에서 config와 학습용 음원 경로를 연결해야 한다. 원본 metadata를 새 split으로 재생성하면 기존 run과 같은 실험이 아니므로 새 실험 ID와 데이터 분할 기록을 남긴다.
