"""Run one resident specialist on a controlled replay; one process/environment per model."""
from __future__ import annotations
import argparse, importlib.util, json, sys, time, hashlib
from pathlib import Path
import numpy as np
from policy_core import SlidingWindowScheduler
from project_io import ProjectFiles, config_experiment_paths, write_json


def import_audio_module(files):
    name = "_pilot_existing_common_audio"
    spec = importlib.util.spec_from_file_location(name, files.locate("common_audio.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", default="pilot_results")
    parser.add_argument("--model", choices=["efficientat", "yamnet", "mobilenetv4"],
                        required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--efficientat-model-name", default="mn04_as")
    args = parser.parse_args()
    files = ProjectFiles(args.project_root)
    out = Path(args.out).resolve()
    config_text = (out / "policy_config.json").read_bytes()
    manifest_text = (out / "replay_manifest.json").read_bytes()
    cfg = json.loads(config_text)
    manifest = json.loads(manifest_text)
    fingerprint = hashlib.sha256(config_text + manifest_text).hexdigest()
    names = cfg["class_names"]
    if files.class_names != names:
        raise ValueError("Class order changed")
    specialist_dir, _ = config_experiment_paths(cfg)
    audio = import_audio_module(files)
    audio_cfg = audio.AudioConfig(sample_rate=int(manifest["sample_rate"]),
                                  clip_seconds=float(manifest["window_seconds"]),
                                  **manifest["audio_config"])
    # Do not silently import a different project's config from the shell cwd.
    sys.path.insert(0, str(files.root))
    if args.model in ("efficientat", "mobilenetv4"):
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu only for CPU profiling")
        device = torch.device(args.device)
        checkpoint_file = files.locate(f"{specialist_dir}/{args.model}/best.pt")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if list(checkpoint.get("class_names", [])) != names:
            raise ValueError("Checkpoint class order differs")
        if args.model == "efficientat":
            from train_efficientat import EfficientATBaseline
            model = EfficientATBaseline(
                files.root / "third_party/EfficientAT", args.efficientat_model_name,
                len(names), freeze_backbone=True,
            )
        else:
            from train_mobilenetv4 import MobileNetV4Baseline
            model = MobileNetV4Baseline(len(names), freeze_backbone=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()

        def infer(wav):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                value = model(torch.from_numpy(wav[None, :]).to(device))
                result = value.detach().cpu().numpy()[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return result, time.perf_counter() - start

        runtime = {"torch": torch.__version__, "device": str(device)}
        if device.type == "cuda":
            runtime["gpu"] = torch.cuda.get_device_name(device)
    else:
        import tensorflow as tf
        import tensorflow_hub as hub
        gpus = tf.config.list_physical_devices("GPU")
        if args.device.startswith("cuda") and not gpus:
            raise RuntimeError("TensorFlow sees no GPU; verify the existing YAMNet environment")
        if args.device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        model = tf.keras.models.load_model(
            files.locate(f"{specialist_dir}/yamnet/best.keras"), compile=False)
        if tuple(model.output_shape)[-1] != len(names):
            raise ValueError("YAMNet head class dimension changed")

        def infer(wav):
            start = time.perf_counter()
            _, embedding, _ = yamnet(tf.convert_to_tensor(wav, dtype=tf.float32))
            pooled = tf.reduce_mean(embedding, axis=0, keepdims=True)
            result = model(pooled, training=False).numpy()[0]
            return result, time.perf_counter() - start

        runtime = {"tensorflow": tf.__version__,
                   "gpu_devices": [g.name for g in tf.config.get_visible_devices("GPU")]}

    # Load/tracing warm-up excluded from per-window service time.
    for _ in range(2):
        infer(np.zeros(audio_cfg.num_samples, dtype=np.float32))
    keys, logits, preprocess_s, inference_s = [], [], [], []
    for episode in manifest["episodes"]:
        # Match existing file loader: mono -> resample -> first 2s/pad.
        raw, sr = audio.sf.read(episode["path"], dtype="float32", always_2d=True)
        raw = audio._to_mono(raw).astype(np.float32, copy=False)
        raw = audio._resample(raw, int(sr), audio_cfg.sample_rate).astype(np.float32)
        raw = audio._fix_length(raw, audio_cfg.num_samples)
        waveform = np.concatenate((
            np.zeros(round(manifest["warmup_seconds"] * audio_cfg.sample_rate), np.float32),
            raw,
            np.zeros(round(manifest["tail_seconds"] * audio_cfg.sample_rate), np.float32),
        ))
        scheduler = SlidingWindowScheduler(
            audio_cfg.sample_rate, manifest["window_seconds"], manifest["base_hop_seconds"])
        # This is offline replay. Each slice contains no samples beyond its end.
        for window in scheduler.push(waveform):
            begin = time.perf_counter()
            wav = audio.highpass_filter(window.waveform, audio_cfg).astype(np.float32)
            dbfs = audio.max_frame_rms_dbfs(wav, audio_cfg)
            if dbfs >= audio_cfg.silence_threshold_dbfs:
                wav = audio.rms_normalize(wav, audio_cfg).astype(np.float32)
            wav = np.ascontiguousarray(wav, dtype=np.float32)
            preprocessing = time.perf_counter() - begin
            z, service = infer(wav)
            if z.shape != (len(names),) or not np.isfinite(z).all():
                raise ValueError("Invalid model output")
            keys.append(f"{episode['episode_id']}:{window.end_sample}")
            logits.append(z)
            preprocess_s.append(preprocessing)
            inference_s.append(service)
        print(f"{args.model}: episode {episode['episode_id']+1}/{len(manifest['episodes'])}",
              flush=True)
    np.savez_compressed(
        out / f"replay_{args.model}.npz",
        keys=np.asarray(keys), logits=np.asarray(logits),
        preprocessing_s=np.asarray(preprocess_s), inference_s=np.asarray(inference_s),
        fingerprint=np.asarray(fingerprint),
    )
    runtime.update({
        "model": args.model, "fingerprint": fingerprint,
        "batch_size": 1, "windows": len(keys),
        "preprocess_ms_p50": float(np.median(preprocess_s) * 1000),
        "model_ms_p50": float(np.median(inference_s) * 1000),
        "service_ms_p95": float(np.quantile(
            np.asarray(preprocess_s) + inference_s, 0.95) * 1000),
        "scope": "Isolated worker replay, not simultaneous 3-model scheduling or actual UI latency.",
    })
    write_json(out / f"runtime_{args.model}.json", runtime)
    print(runtime)


if __name__ == "__main__":
    main()

