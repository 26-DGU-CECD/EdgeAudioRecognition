#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import queue
import sys
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio


SAMPLE_RATE = 16000
MODEL_SAMPLE_RATE = 32000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MODEL_INPUT_SECONDS = 10
MODEL_INPUT_SAMPLES = MODEL_SAMPLE_RATE * MODEL_INPUT_SECONDS
REQUIRED_INPUT_CHANNELS = 6
MIC_CHANNEL_INDEX = 0
MODEL_NAME = "mn10_as"
AUDIOSET_CLASS_COUNT = 527
DB_EPSILON = 1e-12
DEFAULT_MIN_DB = 30.0
DEFAULT_ENHANCE_THRESHOLD_DB = 35.0
DEFAULT_NOISE_REDUCTION_DB = 18.0
DEFAULT_MAIN_GAIN_DB = 8.0
DEFAULT_ENHANCE_SHARPNESS = 2.0
DEFAULT_MIN_SCORE = 0.05

# EfficientAT AudioSet pretrained models use the official 32 kHz frontend.
N_MELS = 128
WINDOW_SIZE = 800
HOP_SIZE = 320
N_FFT = 1024

MIC_NAME_KEYWORDS = ("respeaker", "re speaker", "seeed", "array v3")

LABEL_MAPPING = {
    "construction": ["Tools", "Power tool", "Jackhammer", "Drill", "Chainsaw", "Hammer", "Sawing"],
    "gunshot":      ["Gunshot, gunfire"],
    "alarm_siren":  ["Siren", "Alarm", "Alarm clock"],
    "horn":         ["Vehicle horn, car horn, honking"],
    "water":        [
        "Water",
        "Rain",
        "Raindrop",
        "Rain on surface",
        "Stream",
        "Waterfall",
        "Gurgling",
        "Water tap, faucet",
        "Sink (filling or washing)",
        "Liquid",
        "Splash, splatter",
        "Pour",
    ],
    "knock":        ["Knock"],
    "appliances":   ["Vacuum cleaner"],
    "baby_cry":     ["Baby cry, infant cry"],
    "animal_cry":   ["Dog", "Cat", "Caterwaul"],
    "glass_shatter":["Glass", "Shatter"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime EfficientAT inference from ReSpeaker Array V3."
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(__file__).resolve().parent / "EfficientAT"),
        help="Path to cloned fschmid56/EfficientAT repository.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional sounddevice input device index. Defaults to automatic ReSpeaker search.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=MIC_CHANNEL_INDEX,
        help=(
            "Input channel to use. ReSpeaker 4 Mic Array 6-channel firmware is usually "
            "ch0=processed audio, ch1-4=raw microphones, ch5=playback."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print waveform, mel, logits, and top AudioSet sigmoid scores for each chunk.",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help=(
            "Ignore chunks quieter than this level. Positive values are treated as "
            "dB below full scale, so 30 means -30 dBFS. Use 0 or a negative value "
            "to pass an explicit dBFS threshold."
        ),
    )
    parser.add_argument(
        "--enhance-threshold-db",
        type=float,
        default=DEFAULT_ENHANCE_THRESHOLD_DB,
        help=(
            "Sample-level enhancement threshold. Positive values are treated as "
            "dB below full scale, so 35 means -35 dBFS."
        ),
    )
    parser.add_argument(
        "--noise-reduction-db",
        type=float,
        default=DEFAULT_NOISE_REDUCTION_DB,
        help="Reduce quieter waveform parts by this many dB before inference.",
    )
    parser.add_argument(
        "--main-gain-db",
        type=float,
        default=DEFAULT_MAIN_GAIN_DB,
        help="Boost louder waveform parts by this many dB before inference.",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        dest="main_gain_db",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enhance-sharpness",
        type=float,
        default=DEFAULT_ENHANCE_SHARPNESS,
        help="Higher values separate quiet noise and loud events more aggressively.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Only print predictions whose best custom sigmoid score is at least this value.",
    )
    return parser.parse_args()


def input_devices() -> List[Tuple[int, dict]]:
    return [
        (index, dict(device))
        for index, device in enumerate(sd.query_devices())
        if int(device.get("max_input_channels", 0)) > 0
    ]


def print_input_devices() -> None:
    devices = input_devices()
    if not devices:
        print("사용 가능한 입력 디바이스가 없습니다.")
        return

    print("사용 가능한 입력 디바이스:")
    for index, device in devices:
        print(
            f"  [{index}] {device.get('name')} | "
            f"inputs={device.get('max_input_channels')} | "
            f"default_sr={device.get('default_samplerate')}"
        )


def find_respeaker_device(device_index: int | None) -> Tuple[int, dict, int]:
    devices = input_devices()

    if device_index is not None:
        for index, device in devices:
            if index == device_index:
                channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
                return index, device, channels
        raise RuntimeError(f"입력 디바이스 index {device_index}를 찾을 수 없습니다.")

    candidates = []
    for index, device in devices:
        name = str(device.get("name", "")).lower()
        max_channels = int(device.get("max_input_channels", 0))
        if max_channels <= MIC_CHANNEL_INDEX:
            continue
        if any(keyword in name for keyword in MIC_NAME_KEYWORDS):
            candidates.append((index, device))

    if candidates:
        candidates.sort(
            key=lambda item: (
                int(item[1].get("default_samplerate", 0)) != SAMPLE_RATE,
                int(item[1].get("max_input_channels", 0)) < REQUIRED_INPUT_CHANNELS,
                item[0],
            )
        )
        index, device = candidates[0]
        channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
        return index, device, channels

    raise RuntimeError("ReSpeaker Array V3 입력 디바이스를 자동 탐색하지 못했습니다.")


def load_efficientat(
    efficientat_dir: Path,
    device: torch.device,
):
    if not efficientat_dir.exists():
        raise RuntimeError(
            f"EfficientAT 저장소가 없습니다: {efficientat_dir}\n"
            "먼저 `bash install.sh`를 실행하세요."
        )

    repo_dir = efficientat_dir.resolve()
    sys.path.insert(0, str(repo_dir))

    old_cwd = os.getcwd()
    try:
        # EfficientAT helper modules load metadata/resources by relative path.
        os.chdir(str(repo_dir))
        from helpers.utils import NAME_TO_WIDTH, labels  # type: ignore
        from models.mn.model import get_model as get_mn  # type: ignore
        from models.preprocess import AugmentMelSTFT  # type: ignore

        with redirect_stdout(io.StringIO()):
            model = get_mn(
                width_mult=NAME_TO_WIDTH(MODEL_NAME),
                pretrained_name=MODEL_NAME,
                strides=(2, 2, 2, 2),
                head_type="mlp",
            )

        mel = AugmentMelSTFT(
            n_mels=N_MELS,
            sr=MODEL_SAMPLE_RATE,
            win_length=WINDOW_SIZE,
            hopsize=HOP_SIZE,
            n_fft=N_FFT,
            freqm=0,
            timem=0,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torchvision":
            raise RuntimeError(
                "EfficientAT 모델 로딩에 torchvision이 필요합니다. "
                "`pip install torchvision` 후 다시 실행하세요."
            ) from exc
        raise
    finally:
        os.chdir(old_cwd)

    model.to(device).eval()
    mel.to(device).eval()
    return model, mel, list(labels)


def build_custom_label_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    if len(labels) != AUDIOSET_CLASS_COUNT:
        raise RuntimeError(
            f"AudioSet 라벨 수가 {AUDIOSET_CLASS_COUNT}개가 아닙니다: {len(labels)}"
        )

    label_to_index = {label: index for index, label in enumerate(labels)}
    indices: Dict[str, List[int]] = {}
    missing: Dict[str, List[str]] = {}

    for custom_label, audioset_labels in LABEL_MAPPING.items():
        matched = [label_to_index[label] for label in audioset_labels if label in label_to_index]
        not_found = [label for label in audioset_labels if label not in label_to_index]
        indices[custom_label] = matched
        if not_found:
            missing[custom_label] = not_found

    if missing:
        details = "; ".join(
            f"{custom_label}: {', '.join(labels)}"
            for custom_label, labels in missing.items()
        )
        raise RuntimeError(f"AudioSet 라벨을 찾지 못했습니다: {details}")

    return indices


def predict_chunk(
    waveform: np.ndarray,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    resampler: torch.nn.Module | None,
    custom_indices: Dict[str, List[int]],
    audioset_labels: Sequence[str],
    device: torch.device,
    debug: bool = False,
) -> Tuple[str, float, Dict[str, float]]:
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    input_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    if resampler is not None:
        input_tensor = resampler(input_tensor)
    if input_tensor.shape[1] < MODEL_INPUT_SAMPLES:
        input_tensor = torch.nn.functional.pad(
            input_tensor,
            (0, MODEL_INPUT_SAMPLES - input_tensor.shape[1]),
        )
    elif input_tensor.shape[1] > MODEL_INPUT_SAMPLES:
        input_tensor = input_tensor[:, :MODEL_INPUT_SAMPLES]

    amp_context = torch.cuda.amp.autocast(enabled=True) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_context:
        spec = mel(input_tensor)
        logits, _ = model(spec.unsqueeze(0))
        probabilities = torch.sigmoid(logits.float()).squeeze(0).detach().cpu().numpy()

    if probabilities.shape[0] != AUDIOSET_CLASS_COUNT:
        raise RuntimeError(
            f"모델 출력 클래스 수가 {AUDIOSET_CLASS_COUNT}개가 아닙니다: {probabilities.shape[0]}"
        )

    if debug:
        logits_cpu = logits.float().squeeze(0).detach().cpu().numpy()
        spec_cpu = spec.detach().float().cpu().numpy()
        input_cpu = input_tensor.detach().float().cpu().numpy().squeeze(0)
        top_indices = np.argsort(probabilities)[::-1][:10]
        top_text = ", ".join(
            f"{audioset_labels[index]}={probabilities[index]:.4f}/logit={logits_cpu[index]:+.2f}"
            for index in top_indices
        )
        print(
            "DEBUG "
            f"wav[min={input_cpu.min():+.4f}, max={input_cpu.max():+.4f}, "
            f"rms={np.sqrt(np.mean(np.square(input_cpu))):.6f}] | "
            f"mel[min={spec_cpu.min():+.3f}, max={spec_cpu.max():+.3f}, mean={spec_cpu.mean():+.3f}] | "
            f"logits[min={logits_cpu.min():+.2f}, max={logits_cpu.max():+.2f}, "
            f"mean={logits_cpu.mean():+.2f}] | "
            f"top={top_text}",
            file=sys.stderr,
            flush=True,
        )

    scores = {
        custom_label: float(np.max(probabilities[label_indices]))
        for custom_label, label_indices in custom_indices.items()
    }
    best_label = max(scores, key=scores.get)
    return best_label, scores[best_label], scores


def rms_dbfs(waveform: np.ndarray) -> float:
    waveform = np.asarray(waveform, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    return 20.0 * np.log10(max(rms, DB_EPSILON))


def db_gate_threshold(min_db: float) -> float:
    if min_db > 0:
        return -min_db
    return min_db


def enhance_threshold_amplitude(enhance_threshold_db: float) -> float:
    dbfs = db_gate_threshold(enhance_threshold_db)
    return float(10.0 ** (dbfs / 20.0))


def enhance_chunk(
    waveform: np.ndarray,
    enhance_threshold_db: float,
    noise_reduction_db: float,
    main_gain_db: float,
    enhance_sharpness: float,
) -> Tuple[np.ndarray, bool, float, float]:
    waveform = np.asarray(waveform, dtype=np.float32)
    threshold = enhance_threshold_amplitude(enhance_threshold_db)
    quiet_gain = float(10.0 ** (-abs(noise_reduction_db) / 20.0))
    loud_gain = float(10.0 ** (main_gain_db / 20.0))
    sharpness = max(float(enhance_sharpness), 0.1)

    relative_level = np.abs(waveform) / max(threshold, DB_EPSILON)
    loud_weight = np.power(relative_level, sharpness)
    loud_weight = loud_weight / (1.0 + loud_weight)
    gain = quiet_gain + (loud_gain - quiet_gain) * loud_weight

    enhanced = waveform * gain.astype(np.float32, copy=False)
    clipped = bool(np.any(np.abs(enhanced) > 1.0))
    enhanced = np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)
    return enhanced, clipped, quiet_gain, loud_gain


def format_scores(scores: Dict[str, float]) -> str:
    return ", ".join(f"{label}={probability:.1%}" for label, probability in scores.items())


def run_stream(
    device_index: int,
    device_info: dict,
    stream_channels: int,
    channel_index: int,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    resampler: torch.nn.Module | None,
    custom_indices: Dict[str, List[int]],
    audioset_labels: Sequence[str],
    device: torch.device,
    debug: bool,
    min_db: float,
    enhance_threshold_db: float,
    noise_reduction_db: float,
    main_gain_db: float,
    enhance_sharpness: float,
    min_score: float,
) -> None:
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    if channel_index < 0 or channel_index >= stream_channels:
        raise RuntimeError(
            f"선택한 채널 index가 범위를 벗어났습니다: "
            f"channel={channel_index}, available=0..{stream_channels - 1}"
        )

    if stream_channels < REQUIRED_INPUT_CHANNELS:
        print(
            f"경고: 선택한 디바이스가 {stream_channels}개 입력 채널만 보고합니다. "
            "사용 가능한 채널 0으로 계속 진행합니다.",
            file=sys.stderr,
        )

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"오디오 입력 상태: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print(
        f"입력 디바이스: [{device_index}] {device_info.get('name')} | "
        f"channels={stream_channels}, mic_sr={SAMPLE_RATE}, "
        f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
        f"model_input={MODEL_INPUT_SECONDS}s, channel={channel_index}, "
        f"min_dbfs={db_gate_threshold(min_db):+.1f}, "
        f"enhance_threshold_dbfs={db_gate_threshold(enhance_threshold_db):+.1f}, "
        f"noise_reduction_db={noise_reduction_db:.1f}, main_gain_db={main_gain_db:+.1f}, "
        f"min_score={min_score:.1%}"
    )
    print("Ctrl+C로 종료합니다.")

    pending_blocks: List[np.ndarray] = []
    pending_samples = 0

    with sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=stream_channels,
        dtype="float32",
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            mono = block[:, channel_index].astype(np.float32, copy=True)
            pending_blocks.append(mono)
            pending_samples += mono.shape[0]

            if pending_samples < CHUNK_SAMPLES:
                continue

            joined = np.concatenate(pending_blocks)
            offset = 0
            while joined.shape[0] - offset >= CHUNK_SAMPLES:
                chunk = joined[offset: offset + CHUNK_SAMPLES]
                offset += CHUNK_SAMPLES
                timestamp = datetime.now().strftime("%H:%M:%S")
                chunk_dbfs = rms_dbfs(chunk)
                min_dbfs = db_gate_threshold(min_db)

                if chunk_dbfs < min_dbfs:
                    if debug:
                        print(
                            f"[{timestamp}] skip: level={chunk_dbfs:+.1f} dBFS "
                            f"< gate={min_dbfs:+.1f} dBFS",
                            flush=True,
                        )
                    continue

                inference_chunk, clipped, quiet_gain, loud_gain = enhance_chunk(
                    chunk,
                    enhance_threshold_db,
                    noise_reduction_db,
                    main_gain_db,
                    enhance_sharpness,
                )

                try:
                    best_label, best_probability, scores = predict_chunk(
                        inference_chunk,
                        model,
                        mel,
                        resampler,
                        custom_indices,
                        audioset_labels,
                        device,
                        debug=debug,
                    )
                except Exception as exc:  # Keep the stream alive on bad chunks.
                    print(
                        f"[{timestamp}] 추론 오류: {exc} | 해당 청크 skip",
                        file=sys.stderr,
                    )
                    continue

                if best_probability < min_score:
                    if debug:
                        print(
                            f"[{timestamp}] ignore: best={best_label} "
                            f"({best_probability:.1%}) < min_score={min_score:.1%} | "
                            f"전체: {format_scores(scores)}",
                            flush=True,
                        )
                    continue

                print(
                    f"[{timestamp}] 예측: {best_label} ({best_probability:.1%}) | "
                    f"level={chunk_dbfs:+.1f} dBFS | "
                    f"enhanced={rms_dbfs(inference_chunk):+.1f} dBFS | "
                    f"quiet_gain={quiet_gain:.2f}x loud_gain={loud_gain:.2f}x"
                    f"{' clipped' if clipped else ''} | 전체: {format_scores(scores)}",
                    flush=True,
                )

            remainder = joined[offset:]
            pending_blocks = [remainder] if remainder.size else []
            pending_samples = remainder.shape[0]


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = find_respeaker_device(args.device_index)
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        print_input_devices()
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"추론 디바이스: {device}")
    resampler = None
    if SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    try:
        model, mel, audioset_labels = load_efficientat(Path(args.efficientat_dir), device)
        custom_indices = build_custom_label_indices(audioset_labels)
    except Exception as exc:
        print(f"모델 초기화 오류: {exc}", file=sys.stderr)
        return 1

    try:
        run_stream(
            device_index=device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=args.channel_index,
            model=model,
            mel=mel,
            resampler=resampler,
            custom_indices=custom_indices,
            audioset_labels=audioset_labels,
            device=device,
            debug=args.debug,
            min_db=args.min_db,
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
            min_score=args.min_score,
        )
    except KeyboardInterrupt:
        print("\n종료합니다.")
        return 0
    except Exception as exc:
        print(f"오디오 스트림 오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
