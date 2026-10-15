# Hardcoded-path execution

Edit only `config.py` before running.

```python
PROJECT_ROOT = Path("/absolute/path/to/audio_baselines_3models_v5")
DATASET_ROOT = Path("/absolute/path/to/audio_dataset")
```

The remaining paths are derived in code:

- `CLASS_NAMES_PATH = PROJECT_ROOT / "class_names.txt"`
- `METADATA_CSV = PROJECT_ROOT / "metadata.csv"`
- `EFFICIENTAT_ROOT = PROJECT_ROOT / "third_party/EfficientAT"`
- model outputs under `PROJECT_ROOT / "runs"`

No environment variables are required.

## PyTorch environment

```bash
conda activate audio-torch
cd /absolute/path/to/audio_baselines_3models_v5
pip install -r requirements-torch.txt
mkdir -p third_party
git clone https://github.com/fschmid56/EfficientAT third_party/EfficientAT
python build_metadata.py
python audit_dataset.py
python train_efficientat.py
python train_mobilenetv4.py
```

## YAMNet environment

```bash
conda activate audio-yamnet
cd /absolute/path/to/audio_baselines_3models_v5
pip install -r requirements-yamnet.txt
python train_yamnet.py
```

All scripts still accept optional command-line overrides, but normal execution uses the constants in `config.py`.
