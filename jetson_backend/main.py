#!/usr/bin/env python3
"""Canonical ReSpeaker/WAV -> specialized detectors -> BLE entry point."""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import numpy as np

import runtime_config
from audio_buffer import SlidingWindowBuffer
from audio_level_meter import AudioLevelMeter
from audio_queue import AudioQueue
from audio_stream_controller import AudioStreamController
from cli import parse_args
from db_threshold_gate import DbThresholdGate
from decision import DecisionGate, load_thresholds
from detector_registry import build_detectors
from io_setup import configure_utf8_stdio
from motion_state import MotionMonitor
from parallel_inference import ParallelInferenceEngine


def _load_wav(path: str) -> np.ndarray:
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != runtime_config.SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=runtime_config.SAMPLE_RATE,
        )
    return np.asarray(waveform, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = parse_args(argv)
    runtime_config.apply_path_overrides(
        checkpoints_dir=args.checkpoints_dir,
        thresholds=args.thresholds,
        efficientat_dir=args.efficientat_dir,
    )

    hop_samples = int(round(args.hop_seconds * runtime_config.SAMPLE_RATE))
    try:
        window_buffer = SlidingWindowBuffer(
            runtime_config.WINDOW_SAMPLES,
            hop_samples,
        )
    except ValueError as exc:
        print(f"윈도우 설정 오류: {exc}", file=sys.stderr)
        return 1

    if args.input_wav is not None and not Path(args.input_wav).is_file():
        print(f"입력 파일이 없습니다: {args.input_wav}", file=sys.stderr)
        return 1

    device_index = device_info = stream_channels = None
    if args.input_wav is None:
        try:
            from device_finder import InputDeviceFinder
        except ImportError as exc:
            print(
                f"마이크 입력 모듈을 불러올 수 없습니다: {exc}\n"
                "Pi에서는 portaudio19-dev 설치 후 sounddevice를 설치하세요. "
                "하드웨어 없이 확인하려면 --input-wav를 사용하세요.",
                file=sys.stderr,
            )
            return 1

        finder = InputDeviceFinder()
        if args.list_devices:
            finder.print_input_devices()
            return 0
        try:
            device_index, device_info, stream_channels = finder.find_respeaker_device(
                args.device_index
            )
        except Exception as exc:
            print(f"마이크 장치 오류: {exc}", file=sys.stderr)
            finder.print_input_devices()
            return 1
    elif args.list_devices:
        print("--input-wav와 --list-devices는 함께 사용할 수 없습니다.", file=sys.stderr)
        return 1

    ble_server = None
    if not args.no_ble:
        try:
            from ble_inference_server import BleInferenceServer

            ble_server = BleInferenceServer(args.ble_name, args.ble_chunk_bytes)
            ble_server.start()
        except Exception as exc:
            print(
                f"BLE 초기화 오류: {exc}\n"
                "추론만 확인하려면 --no-ble을 사용하세요.",
                file=sys.stderr,
            )
            return 1

    engine = None
    audio_queue = AudioQueue(
        runtime_config.QUEUE_MAX_SECONDS,
        runtime_config.SAMPLE_RATE,
    )
    try:
        thresholds = load_thresholds(runtime_config.THRESHOLDS_JSON)
        detectors = build_detectors(
            args.detector_names,
            batch_size=1,
            torch_threads=args.torch_threads,
        )
        engine = ParallelInferenceEngine(
            detectors,
            concurrent=args.concurrent,
        )
        decision_gate = DecisionGate(
            thresholds,
            classes=engine.classes,
            debounce_seconds=args.debounce_seconds,
        )
    except Exception as exc:
        print(f"모델/threshold 초기화 오류: {exc}", file=sys.stderr)
        if ble_server is not None:
            ble_server.stop()
        return 1

    microphone = None
    if args.input_wav is None:
        from microphone_module import MicrophoneModule

        microphone = MicrophoneModule(
            device_index=device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=args.channel_index,
            audio_queue=audio_queue,
        )

    imu = None
    if args.imu:
        try:
            from imu_module import ImuError, open_imu

            imu = open_imu(
                bus=args.imu_bus,
                address=args.imu_address,
                sample_hz=args.imu_sample_hz,
                history_seconds=max(4.0, runtime_config.WINDOW_SECONDS * 2.0),
            )
        except (ImportError, ImuError) as exc:
            print(
                f"IMU 초기화 오류: {exc}\n"
                "IMU 없이 실행하려면 --imu를 빼고 다시 실행하세요. "
                "연결 확인은 `python3 test_imu.py --scan`을 쓰세요.",
                file=sys.stderr,
            )
            if engine is not None:
                engine.close()
            if ble_server is not None:
                ble_server.stop()
            return 1

    motion_monitor = MotionMonitor(
        imu,
        window_seconds=runtime_config.WINDOW_SECONDS,
        suppress_on_motion=args.suppress_on_motion,
    )

    controller = AudioStreamController(
        audio_queue=audio_queue,
        window_buffer=window_buffer,
        level_meter=AudioLevelMeter(),
        threshold_gate=DbThresholdGate(args.min_db),
        inference_engine=engine,
        decision_gate=decision_gate,
        microphone=microphone,
        publisher=ble_server,
        motion_monitor=motion_monitor,
        skip_low_db=args.skip_low_db,
        debug=args.debug,
        max_windows_per_cycle=runtime_config.MAX_WINDOWS_PER_CYCLE,
    )

    def stop(_signum, _frame) -> None:  # noqa: ANN001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    exit_code = 0
    try:
        if args.input_wav is not None:
            processed = controller.run_file(_load_wav(args.input_wav))
            print(f"처리된 윈도우: {processed}")
        else:
            controller.run()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    except Exception as exc:
        print(f"오디오 스트림 오류: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if audio_queue.total_dropped_blocks or controller.dropped_windows:
            print(
                f"폐기 누계: 오디오 블록 {audio_queue.total_dropped_blocks}, "
                f"윈도우 {controller.dropped_windows}",
                file=sys.stderr,
            )
        if controller.motion_suppressed_windows:
            print(
                f"움직임으로 건너뛴 윈도우: {controller.motion_suppressed_windows}",
                file=sys.stderr,
            )
        if imu is not None:
            if imu.read_errors:
                print(f"IMU 읽기 오류 누계: {imu.read_errors}", file=sys.stderr)
            imu.stop()
        if engine is not None:
            engine.close()
        if ble_server is not None:
            ble_server.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
