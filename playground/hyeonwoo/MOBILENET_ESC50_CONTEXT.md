# MobileNetV3 ESC-50 Subset Workflow Context

This file is the handoff context for continuing this work from another computer
or another Codex session.

## Goal

Train a lightweight sound classification model on a filtered ESC-50 subset, then
deploy the trained checkpoint or ONNX model to NVIDIA Jetson Orin Nano.

The Jetson is intended mainly for inference/deployment tests. Training should be
done on a laptop/desktop when possible, then the trained artifacts should be
copied to the Jetson.

## Main Script

Use:

```bash
playground/hyeonwoo/esc50_mobilenet_workflow.py
```

The script supports:

```bash
python esc50_mobilenet_workflow.py train
python esc50_mobilenet_workflow.py export
python esc50_mobilenet_workflow.py infer --wav path/to/test.wav
```

The script defaults all paths to the directory where the script is located:

```text
EdgeAudioRecognition/playground/hyeonwoo/
```

Generated artifacts:

```text
playground/hyeonwoo/best_model.pth
playground/hyeonwoo/label_map.json
playground/hyeonwoo/model.onnx
```

## Target Classes

The final consolidated target classes are:

```text
construction
gunshot
alarm_siren
horn
water
knock
appliances
baby_cry
animal_cry
glass_shatter
```

ESC-50 category mapping:

```python
{
    "construction": ["jackhammer", "drilling"],
    "gunshot": ["gunshot"],
    "alarm_siren": ["siren", "clock_alarm"],
    "horn": ["car_horn"],
    "water": ["rain", "pouring_water", "water_drops"],
    "knock": ["door_wood_knock"],
    "appliances": ["washing_machine", "vacuum_cleaner"],
    "baby_cry": ["crying_baby"],
    "animal_cry": ["dog", "cat"],
    "glass_shatter": ["glass_shattering"],
}
```

`bicycle` and `human scream` were intentionally omitted because ESC-50 has no
direct category match.

## Dataset Layout

Expected dataset location:

```text
playground/hyeonwoo/ESC-50-master/
  audio/
    *.wav
  meta/
    esc50.csv
```

On the Jetson, the dataset was downloaded and extracted here:

```text
/home/bugless/EdgeAudioRecognition/playground/hyeonwoo/ESC-50-master
```

If setting up on a new PC:

```bash
cd EdgeAudioRecognition/playground/hyeonwoo
wget https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip -O ESC-50-master.zip
unzip ESC-50-master.zip
```

## PC Training Setup

Recommended: train on PC, not Jetson.

From the repo root or from `playground/hyeonwoo`:

```bash
cd EdgeAudioRecognition/playground/hyeonwoo
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio numpy==1.26.4 onnx onnxruntime
```

Train:

```bash
python esc50_mobilenet_workflow.py train --epochs 40 --batch-size 32 --val-fold 5
```

If GPU memory is limited:

```bash
python esc50_mobilenet_workflow.py train --epochs 40 --batch-size 8 --grad-accum-steps 4 --val-fold 5
```

Export ONNX:

```bash
python esc50_mobilenet_workflow.py export
```

## Jetson Runtime Setup

Jetson environment used:

```text
Jetson Orin Nano
JetPack/L4T 36.4.7
CUDA 12.6
torch 2.8.0
torchvision 0.23.0
torchaudio 2.8.0
numpy 1.26.4
```

Jetson-compatible install commands used:

```bash
cd ~/EdgeAudioRecognition/playground/hyeonwoo
source venv/bin/activate

pip install numpy==1.26.4
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install torchaudio==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Important Jetson note:

Generic PyTorch wheels may detect CUDA but fail on Orin with:

```text
CUDA error: no kernel image is available for execution on the device
```

Use Jetson-compatible wheels from the Jetson AI Lab index.

## Copy Artifacts From PC To Jetson

After PC training and ONNX export:

```bash
scp best_model.pth label_map.json model.onnx \
  bugless@JETSON_IP:/home/bugless/EdgeAudioRecognition/playground/hyeonwoo/
```

Then on Jetson:

```bash
cd ~/EdgeAudioRecognition/playground/hyeonwoo
source venv/bin/activate
python esc50_mobilenet_workflow.py infer --wav /path/to/test.wav
```

## Model Details

Backbone:

```text
torchvision.models.mobilenet_v3_small
```

Input:

```text
1-channel log-mel spectrogram
```

Default audio preprocessing:

```text
sample_rate: 16000
duration: 5.0 sec
n_fft: 1024
hop_length: 320
n_mels: 64
```

The first MobileNet convolution is changed from 3 input channels to 1 input
channel. The final classifier output dimension is changed to 10 classes.

## Known Jetson Training Issue

Jetson training can hit CUDA OOM even with a lightweight model because training
stores activations, gradients, and optimizer state. Prefer PC training. If
training on Jetson is unavoidable, use a small batch and gradient accumulation:

```bash
python esc50_mobilenet_workflow.py train --epochs 40 --batch-size 2 --grad-accum-steps 8 --val-fold 5
```

## What To Tell A New Codex Session

Ask the new session to read this file first:

```text
playground/hyeonwoo/MOBILENET_ESC50_CONTEXT.md
```

Suggested prompt:

```text
Read playground/hyeonwoo/MOBILENET_ESC50_CONTEXT.md first. I am training the
MobileNetV3 ESC-50 subset classifier on my PC and deploying best_model.pth /
model.onnx to Jetson Orin Nano. Continue from that context.
```
