# D / E / F 실행 순서

## 전제

- A EfficientAT, B YAMNet, C MobileNetV4가 `runs/` 아래에 학습되어 있어야 합니다.
- 모델 선택, 전문 클래스 선택, 앙상블 가중치 산출에는 **validation만 사용**합니다.
- test는 마지막 성능 평가에만 사용합니다.
- 클래스 순서는 `class_names.txt`와 동일해야 합니다.

## 0. 경로

`config.py`에서 두 경로를 수정합니다.

```python
PROJECT_ROOT = Path("/home/plass/YAMNet/audio_experiments_D_E_F_v6")
DATASET_ROOT = Path("/home/plass/datasets/environment_audio")
```

기존 v5의 `metadata.csv`, `runs/`, `third_party/EfficientAT`을 새 프로젝트로 복사할 수 있습니다.

```bash
cp /기존/v5/metadata.csv .
cp -a /기존/v5/runs .
mkdir -p third_party
cp -a /기존/v5/third_party/EfficientAT third_party/
```

## 1. 기존 A/B/C에서 validation prediction 내보내기

새 버전으로 다시 학습한 경우 자동 생성되므로 이 단계는 생략합니다.

### audio-torch

```bash
conda activate audio-torch
cd /home/plass/YAMNet/audio_experiments_D_E_F_v6

python export_torch_predictions.py --model efficientat
python export_torch_predictions.py --model mobilenetv4
```

### audio-yamnet

```bash
conda activate audio-yamnet
cd /home/plass/YAMNet/audio_experiments_D_E_F_v6

export TF_ENABLE_ONEDNN_OPTS=0
python export_yamnet_predictions.py
```

각 run 폴더에 다음 파일이 생성됩니다.

- `val_predictions.npz`
- `val_report.json`
- `val_confusion_matrix.png`
- `test_predictions.npz`
- `test_confusion_matrix.png`

## 2. D: 세 모델 전체 데이터 앙상블

A/B/C의 test softmax 확률을 1/3씩 평균합니다.

```bash
conda activate audio-torch
python ensemble_full.py
```

결과:

```text
runs/D_full_ensemble/
├── test_metrics.json
├── test_report.json
├── test_predictions.npz
├── test_confusion_matrix.png
└── test_confusion_matrix_normalized.png
```

## 3. E: 전문 데이터 생성

기본 자동 정책:

- 각 foreground 클래스는 baseline validation F1이 가장 높은 모델에 전문 클래스로 배정
- 전문 클래스 100%
- 해당 모델에서 전문 클래스로 혼동되는 상위 3개 hard-negative 클래스 60%
- 나머지 비전문 클래스 30%
- background는 데이터 풀에 유지하고 batch sampler가 정확히 20% 사용
- train만 축소하며 val/test는 전체 자연 분포 유지
- 샘플 축소는 `group_id` 단위로 수행

```bash
conda activate audio-torch
python build_specialist_metadata.py
```

결과:

```text
specialist_metadata/
├── efficientat.csv
├── yamnet.csv
├── mobilenetv4.csv
├── specialist_assignment.json
├── specialist_sampling_summary.csv
└── baseline_validation_per_class_f1.csv
```

전문 클래스를 직접 지정하려면 `config.py`의 다음 딕셔너리를 채웁니다. 모든 foreground 클래스가 정확히 한 번씩 들어가야 합니다.

```python
SPECIALIST_MANUAL_EXPERT_CLASSES = {
    "efficientat": [],
    "yamnet": [],
    "mobilenetv4": [],
}
```

## 4. E: 세 전문 모델 학습

### audio-torch

```bash
conda activate audio-torch

python train_efficientat_specialist.py
python train_mobilenetv4_specialist.py
```

### audio-yamnet

```bash
conda activate audio-yamnet

export TF_ENABLE_ONEDNN_OPTS=0
python train_yamnet_specialist.py
```

YAMNet 전문 모델은 기존 B 모델의 embedding cache를 재사용하므로 오디오 전체를 다시 임베딩하지 않습니다.

## 5. F: 전문 모델 가중 앙상블

클래스별 모델 가중치는 specialist validation per-class F1로 계산합니다.

```bash
conda activate audio-torch
python ensemble_specialists.py
```

가중치:

```text
weight(model, class)
= max(validation F1, floor)^power
× expert model boost
```

가중치는 클래스별로 세 모델 합이 1이 되도록 정규화됩니다.

결과:

```text
runs/F_specialist_weighted_ensemble/
├── classwise_model_weights.csv
├── specialist_validation_per_class_f1.csv
├── weighting_config.json
├── test_metrics.json
├── test_report.json
├── test_predictions.npz
├── test_confusion_matrix.png
└── test_confusion_matrix_normalized.png
```

## 6. 전체 결과 비교

```bash
python compare_all_results.py
```

## 실험상 주의

- 전문 클래스와 가중치는 test 성능을 보고 결정하지 않습니다.
- `metadata.csv`, 전처리 값, split 또는 오디오가 바뀌면 A/B/C validation prediction을 다시 생성합니다.
- YAMNet 전처리 값이 바뀌면 `embedding_cache`를 삭제하고 다시 생성합니다.
- F의 클래스별 가중치는 specialist validation 데이터만 사용합니다.
