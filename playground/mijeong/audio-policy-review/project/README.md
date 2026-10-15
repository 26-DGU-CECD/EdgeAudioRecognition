# Three audio baseline models — source-group-aware version

A. EfficientAT full-dataset baseline  
B. YAMNet full-dataset baseline  
C. MobileNetV4-Small full-dataset baseline

## Fixed classes

The included `class_names.txt` uses this exact order:

```text
baby_cry
background
bicycle
car_horn
cat_meow
cry
dog_bark
fire_alarm
glass_breaking
gun
knock
scream
siren
water
```

The training label is inferred from the class directory. Audio filenames do not need to be normalized.

## Supported dataset layouts

Already split:

```text
dataset/
  train/
    baby_cry/*.wav
    background/*.wav
    ...
  val/
    ...
  test/
    ...
```

Or class folders only:

```text
dataset/
  baby_cry/*.wav
  background/*.wav
  ...
```

The second layout is automatically split 80/10/10 by `group_id`, not by file. By default, the builder removes the trailing `orig`/`original`/`aug` marker and following numbers from the filename. Files from the same source therefore stay in one split. Review `group_id_audit.csv` before training. Use `--group-regex` when the local naming rule differs.

Background may contain subfolders or filenames that indicate its original class. That information is written to `source_class` for auditing only. Its training target remains `background`.

## 1. Build metadata automatically

```bash
python build_metadata.py \
  --root /path/to/dataset \
  --classes class_names.txt \
  --output metadata.csv
```

The default parser supports names such as `source_orig.wav`, `source_orig_03.wav`, `source_01_aug_2.wav`. When the naming rule differs, use a regex. Example:

```bash
python build_metadata.py \
  --root /path/to/dataset \
  --classes class_names.txt \
  --output metadata.csv \
  --group-regex '(.+?)_(?:segment|chunk)_?\d+$'
```

Generated columns:

```text
path,label,split,source_class,source_status,group_id
```

`source_class` is not consumed by the model. It is retained to analyze where background samples originated. `group_id` is independent of `label`, so an audible target clip and a quiet clip moved to `background` can remain in the same split.

`group_id_audit.csv` summarizes the number of clips, labels and split assigned to each source group. Any group appearing in multiple pre-existing split folders causes an error.

## 2. Audit the existing background rule

The runtime order is high-pass -> maximum 100 ms RMS gate -> non-silent loudness normalization. Audit the labels under the exact same settings:

```bash
python audit_dataset.py \
  --csv metadata.csv \
  --output dataset_audit.csv \
  --highpass-hz 80 \
  --silence-threshold-dbfs -30 \
  --target-rms-dbfs -20
```

This does not relabel files automatically. It identifies:

- non-background clips judged silent by the current pipeline
- background clips judged non-silent by the current pipeline

If the original background migration used a different filter/order/threshold, review these mismatches before training.

## 3. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/fschmid56/EfficientAT third_party/EfficientAT
```

## 4. Train A/B/C

```bash
python train_efficientat.py \
  --csv metadata.csv --classes class_names.txt \
  --efficientat-root third_party/EfficientAT \
  --model-name mn04_as \
  --output runs/A_efficientat --batch-size 40

python train_yamnet.py \
  --csv metadata.csv --classes class_names.txt \
  --output runs/B_yamnet --batch-size 40

python train_mobilenetv4.py \
  --csv metadata.csv --classes class_names.txt \
  --output runs/C_mobilenetv4 --batch-size 40
```

Batch size 40 gives exactly 8 background + 32 non-background samples per training batch. Training sampling is source-group-aware: the sampler chooses a source group first and then one clip from that group, preventing a recording with many `orig`/`aug` derivatives from dominating the gradient. Validation and test retain their natural distribution.

## Important

- The correct silence threshold notation is `-30 dBFS`.
- Do not use the background origin class or `group_id` as a model input or target label.
- Same-origin information is used for split integrity and group-balanced sampling only.
- Do not apply same-group consistency loss across different labels; one source may contain both a target clip and a quiet `background` clip.
- Do not rename files merely for model training; folder-derived labels are sufficient.
- Filename normalization matters only when trying to recover source-recording groups or background provenance automatically.
- Delete YAMNet `embedding_cache/` whenever metadata or preprocessing parameters change.

## Important: separate Python environments

The baseline scripts use two native ML runtimes. To avoid TensorFlow/PyTorch shared-library conflicts, use separate environments:

```bash
python3.11 -m venv .venv-torch
source .venv-torch/bin/activate
pip install -U pip
pip install -r requirements-torch.txt
# Run EfficientAT and MobileNetV4 scripts here.
```

```bash
python3.11 -m venv .venv-yamnet
source .venv-yamnet/bin/activate
pip install -U pip
pip install -r requirements-yamnet.txt
# Run train_yamnet.py here.
```

`train_yamnet.py` imports metadata helpers from `common_metadata.py` and no longer imports PyTorch indirectly.
Do not verify all frameworks in one `python -c` process. Verify each environment independently.
