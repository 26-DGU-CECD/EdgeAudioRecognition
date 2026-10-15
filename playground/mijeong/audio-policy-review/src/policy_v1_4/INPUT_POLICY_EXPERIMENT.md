# 입력 정책 실험과 무투표 specialist 연결 v1.3

## 1. 구현 상태 정정

이전 코드에는 W2초·H2/1/0.5초 입력 스케줄러가 있었지만,
v1.2의 새 무투표 선택 실험은 정적 저장 예측까지만 연결되어 있었다.
기존 evaluate_replay.py는 여전히 가중 fusion을 사용했다.
v1.3에서는 prepare/export에 --selection을 추가하고
evaluate_selection_replay.py를 신설해 원음 window → 각 specialist → 담당 라벨 max 경로를 연결했다.
live 마이크 수집·실제 비동기 worker 운영·실시간 UI는 구현한 것으로 주장하지 않는다.

## 2. 첫 입력 비교군 3개

| 입력 정책 | 문맥 W | 갱신 간격 H | 겹침 | 정상 상태의 전체 모델 호출/음원 1초 | 호출 비율 |
|---|---:|---:|---:|---:|---:|
| I0 기존 | 2초 | 2초 | 0% | 1.5회 | 1배 |
| I1 | 2초 | 1초 | 50% | 3회 | 2배 |
| I2 | 2초 | 0.5초 | 75% | 6회 | 4배 |

모든 입력 정책은 세 specialist를 계속 사용한다.
예: I2의 관찰 창은 [0,2], [0.5,2.5], [1,3]초다.
실제 입력 버퍼에서는 해당 창의 끝까지 수신된 샘플만 사용한다.
2초 학습 데이터를 유지하면서 더 자주 최근 2초를 판단하는 실험이다.
0.5초 입력을 기존 head에 그대로 넣거나, 아직 들어오지 않은 미래 2초를 가정하는 실험이 아니다.
첫 출력은 버퍼 초기 2초 확보 뒤에만 가능하다.
연속 운용 중에는 이미 확보한 과거 문맥이 있으므로 이후 새 입력마다 2초 전체를 다시 기다리는 구조와 구분한다.

S0(담당 라벨 전체 max)를 고정해 입력 효과를 먼저 본다.
모델 checkpoint·라벨 배정·전처리·background 처리 조건을 동일하게 유지한다.
기존 background 경로는 확인되지 않았으므로 all 옵션에서는 각 처리 가정 안에서 I0/I1/I2를 비교한다.
현재 세 정책은 deterministic 고정 주기이며 라벨별 또는 입력 적응형 주기가 아니다.

## 3. 전달받은 S02 결과를 반영한 순서

사용자가 붙여 넣은 S02 설명에서는 global_max 기준 5개 seed 평균이 다음과 같다.
원본 S02 파일을 이 환경에서 직접 검증한 결과와 구분한다.

| 정책 | 출력 정밀도 | 사건 정답 출력률 | 배경 오출력률 | 사건 macro F1 |
|---|---:|---:|---:|---:|
| S0 | 82.71% | 79.51% | 14.42% | 79.40% |
| S1 | 95.10% | 57.42% | 4.52% | 63.74% |
| S2 | 95.20% | 57.21% | 4.25% | 61.76% |

S2가 안정적으로 우수하다고 채택할 근거가 없고,
fire_alarm의 5 seed null·recall0, knock의 대부분 null, baby_cry recall 변동이 보고되었다.
따라서 S2를 입력 실험의 기본 확정 규칙으로 두지 않는다.
추가 seed 반복보다 먼저 원음 연결·실제 선택 구현의 대조를 진행한다.
null은 현재 precision/support 조건을 만족하는 threshold를 못 찾았다는 뜻이지,
해당 라벨의 출력 금지가 검증된 최적 운영 정책이라는 뜻이 아니다.

진행 순서:

1. 평가 그룹에서 원음 manifest 구성 및 경로 대응 확인.
2. CPU에서 파일 열기·길이·sample rate·채널·첫 2초 유한값 확인.
3. fire_alarm–siren 오분류 clip의 청취와 모델별 점수 확인. 오류 혼동 방향은 실제 원음을 보고 확정.
4. 실제 무투표 선택 코드가 확보되면 동일 입력의 라벨·점수·background 처리 대조.
5. S0 고정 I0/I1/I2 신규 window 추론과 비교.
6. 후보를 좁힌 뒤에만 출력 정책 S1/S2를 추가한 교차 비교.

## 4. 원음 회귀 실험 설계

현재 모델은 2초 clip으로 학습했다. 기존 2초 clip 저장 logits만으로
0.5초/1초 이동한 window 결과를 재구성할 수 없으므로 **원음 신규 추론이 필수**다.
원음이 있는 기존 GPU 환경에서 아래 단계를 실행한다.
현재 로컬에서는 원본 데이터셋과 GPU 모델 실행 환경이 없어 음향 정확도나 실제 지연을 측정하지 않았다.

각 recorded group에서 clip 하나를 선택하고 동일 clip의 삽입 위치를 0/0.5/1/1.5초 이동한다.
episode는 `(2초 + 삽입 offset) zero silence + 원음 첫 2초 + 2초 zero silence`다.
이로써 H2초와 사건 clip의 경계가 매번 일치하는 편향을 줄인다.
동일 clip의 여러 offset은 독립 음원 표본으로 세지 않고 group 수를 함께 보고한다.
실제 사건 onset은 모르므로 offset은 **clip 시작 위치**다.
zero silence는 자연 배경·혼합음의 대체물이 아니며 본 실험의 통제 조건이다.

같은 0.5초 기본 격자에서 모델별 예측을 한 번 생성한 뒤
H1초/H2초에 해당하는 창만 추출한다. 두 번의 모델 재학습이 필요한 실험이 아니다.
전처리는 각 실제 창에서 다시 수행하고 YAMNet embedding도 창마다 다시 추출한다.
기존 전처리의 window별 high-pass/RMS 정규화 때문에 겹친 임베딩을 단순 재사용하지 않는다.

## 5. 실행 방법

압축을 푼 코드 폴더에서 실행한다. 처음에는 --max-per-class 1로 연결을 확인한다.
PILOT_PROJECT는 원본 학습·추론 코드와 checkpoint가 합쳐진 폴더다.
PILOT_OUT은 이번 실험만의 새로운 결과 폴더다.
아래 predefined/0.70 기본 실행은 제공된 저장 모델의 **참고 적용 실험**이며 실제 최고 성능 run으로 확인된 것은 아니다.
최고 성능 run은 selection spec에 checkpoint_files·head 순서·저장 예측 경로를 정확히 지정한다.

```bash
conda activate audio-torch
PILOT_PROJECT=/home/plass/YAMNet/audio_experiments_DEF
PILOT_OUT="$PILOT_PROJECT/input_selection_v1_3"

python run_selection_pilot.py \
  --project-root "$PILOT_PROJECT" \
  --background-mode all \
  --out "$PILOT_OUT"

python prepare_replay.py --selection \
  --out "$PILOT_OUT" --max-per-class 1

python check_replay_audio.py --out "$PILOT_OUT"

python export_replay_predictions.py --selection \
  --project-root "$PILOT_PROJECT" --out "$PILOT_OUT" \
  --model efficientat --device cuda

python export_replay_predictions.py --selection \
  --project-root "$PILOT_PROJECT" --out "$PILOT_OUT" \
  --model mobilenetv4 --device cuda

conda activate audio-yamnet
python export_replay_predictions.py --selection \
  --project-root "$PILOT_PROJECT" --out "$PILOT_OUT" \
  --model yamnet --device cuda

python evaluate_selection_replay.py \
  --out "$PILOT_OUT" --background-mode all --output-policies S0
```

기존 Windows GPU 환경에서 PowerShell로 실행할 때는 환경을 활성화한 후:

```powershell
$pilotProject = 'D:\실제\audio_experiments_DEF'
$pilotOut = 'D:\실험결과\input_selection_v1_3'

python run_selection_pilot.py --project-root $pilotProject --background-mode all --out $pilotOut
python prepare_replay.py --selection --out $pilotOut --max-per-class 1 --old-dataset-root /home/plass/YAMNet/aug_data --dataset-root 'D:\실제\aug_data'
python check_replay_audio.py --out $pilotOut
python export_replay_predictions.py --selection --project-root $pilotProject --out $pilotOut --model efficientat --device cuda
python export_replay_predictions.py --selection --project-root $pilotProject --out $pilotOut --model mobilenetv4 --device cuda
# 기존 YAMNet 실행 환경으로 전환한 뒤:
python export_replay_predictions.py --selection --project-root $pilotProject --out $pilotOut --model yamnet --device cuda
python evaluate_selection_replay.py --out $pilotOut --background-mode all --output-policies S0
```

서로 다른 환경이라도 같은 코드 폴더·PILOT_OUT을 사용한다.
Windows 경로가 없는 현재 세션에서 가상의 데이터 경로를 실제 경로처럼 확정하지 않는다.
metadata 경로가 그대로 유효하면 dataset-root 관련 옵션은 생략한다.
파일 연결 검사가 실패하면 audio_preflight.csv에서 실패한 파일을 확인한 뒤 GPU 추론 전에 해결한다.
새 무투표 exporter는 현재 manifest에 대한 audio preflight 통과 기록이 없으면 GPU 로드 전에 중단한다.
원음이나 manifest·정책 설정을 바꾸면 세 모델의 replay export를 모두 다시 생성한다.
이전 v1.2 결과에는 selection_samples.npz와 checkpoint_files가 없을 수 있으므로 새 폴더에서 static 단계부터 실행한다.

S0 입력 결과를 확인한 후에만 선택적으로:

```bash
python evaluate_selection_replay.py \
  --out "$PILOT_OUT" --background-mode global_max --output-policies S0 S1 S2
```

global_max는 실제 기존 background 동작으로 확인된 값이 아니라 하나의 비교 가정이다.
S1/S2의 정적 clip threshold를 그대로 사용하는 조합은 정책 이전 실험이며
rolling window에서 목표 정밀도를 유지한다는 보장이 아니다.

## 6. 평가 결과의 해석

- input_selection_comparison.csv: I0/I1/I2별 clip 삽입 후 2초 내 정답 출력률·미검출 수,
  검출된 사건의 첫 정답까지 audio-time p50/p95, 잘못된 라벨 출력, 배경 clip 알림률,
  prefix zero-silence 알림률, 실행 창·모델 호출 수.
- input_selection_per_episode.csv: 라벨·원본 group·삽입 offset별 결과. fire_alarm/siren 등 취약 라벨을 별도 점검한다.
- input_isolated_worker_cost.csv: 별도 실행한 모델의 window당 전처리+embedding+head 서비스 시간과
  평균 서비스 시간/H. 이는 공유 GPU에서의 실제 점유율·큐 지연이 아니다.
- input_selection_scope.json: 실험 범위, checkpoint hash와 제한사항.

H를 줄이면 결과를 확인할 기회는 늘지만 **정답 확정이 빨라진다는 보장은 없다**.
모델 실행 횟수가 늘어 실제 장치의 큐가 쌓이면 완료 지연이 오히려 증가할 수 있다.
따라서 2초 내 정답 비율·누락·오경보·처리 비용을 함께 비교한다.
검출된 사건의 지연만 보고 미검출을 제외한 채 좋은 정책으로 선정하지 않는다.
본 replay의 audio-time에는 추론 완료·작업 큐·UI 지연이 포함되지 않는다.
실제 장치의 2초 목표 검증은 연속 원음+실제 onset 주석+동시 세 모델 실행 로그로 별도 측정한다.

## 7. 라벨 특성 기반 입력 정책은 후속 단계

이번 구현은 고정 W2초/H2·1·0.5초 비교까지다.
라벨별 문맥 W0.5/1/2초 또는 반복/지속에 따라 추가 관찰을 기다리는 정책은 구현 완료로 표시하지 않는다.
관찰 문맥을 줄이면 학습·전처리·embedding 분포가 달라져 길이별 검증 또는 재학습이 필요하다.
이후에는 같은 라벨 담당을 유지하면서 문맥만 바꾸는 실험으로 분리한다.
최근 예측만 보고 입력 주기를 늦추는 방식은 새로운 단발 위험음을 놓칠 수 있으므로
최대 실행 간격·놓친 사건 수·최악 지연까지 포함한 별도 정책으로 검증한다.
모델 내부 층 계층화는 이번 구현에 넣지 않는다.

## 8. 검증 범위

총 28개 코드 테스트가 통과했다. 추가 통합 검증은 알려진 인공 점수로
창 정렬·삽입 offset·모델별 예측 파일 행 순서와 무관한 점수 정렬·
recorded group 보존·오래된 export 거부를 검사한다.
원음 연결 검사 오류와 현재 manifest의 검사 통과 여부를 GPU 로드 전에 확인하는 동작도 검사했다.
인공 점수 테스트는 음향 분류 성능이나 실제 하드웨어 속도의 증거가 아니다.
