#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys

from realtime_inference import SAMPLE_RATE, print_input_devices

from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from audio_stream_controller import AudioStreamController
from ble_inference_server import AppBleInferenceServer, install_app_compatible_ble_characteristic
from cli import parse_args
from db_threshold_gate import DbThresholdGate
from doa_audio_estimator import AudioDOAEstimator
from doa_selector import DOASelector
from doa_usb_reader import DOAReader
from io_setup import configure_utf8_stdio
from microphone_module import MicrophoneModule
from model_inference import ModelInferenceEngine
from packet_builder import AppSoundPacketBuilder


def main() -> int:
    configure_utf8_stdio()
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    audio_queue = AudioQueue()
    try:
        microphone = MicrophoneModule.auto_detect(
            device_index=args.device_index,
            channel_index=args.channel_index,
            audio_queue=audio_queue,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        print_input_devices()
        return 1

    install_app_compatible_ble_characteristic()
    ble_server = AppBleInferenceServer(args.ble_name, args.ble_chunk_bytes)
    try:
        ble_server.start()
    except Exception as exc:
        print(f"BLE startup error: {exc}", file=sys.stderr, flush=True)
        return 1

    model_engine = ModelInferenceEngine(args.efficientat_dir, debug=args.debug)
    try:
        model_engine.load()
    except Exception as exc:
        print(f"Model initialization error: {exc}", file=sys.stderr, flush=True)
        ble_server.stop()
        return 1

    usb_doa_reader = DOAReader(
        enabled=not args.disable_doa and args.doa_source in ("auto", "usb"),
        poll_interval=args.doa_poll_interval,
        disabled_reason=(
            "disabled by --disable-doa"
            if args.disable_doa
            else "disabled because --doa-source=audio"
        ),
    )
    audio_doa_estimator = AudioDOAEstimator(
        enabled=not args.disable_doa and args.doa_source in ("auto", "audio"),
        stream_channels=microphone.stream_channels,
        sample_rate=SAMPLE_RATE,
        min_db=args.audio_doa_min_db,
        window_ms=args.audio_doa_window_ms,
    )

    controller = AudioStreamController(
        microphone=microphone,
        audio_queue=audio_queue,
        level_meter=AudioLevelMeter(),
        threshold_gate=DbThresholdGate(args.min_db),
        preprocessor=AudioPreprocessor(
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
        ),
        model_engine=model_engine,
        doa_selector=DOASelector(
            doa_source=args.doa_source,
            usb_reader=usb_doa_reader,
            audio_estimator=audio_doa_estimator,
        ),
        ble_server=ble_server,
        packet_builder=AppSoundPacketBuilder(
            north_offset=args.north_offset,
            db_offset=args.db_offset,
            full_packet=args.full_packet,
        ),
        min_score=args.min_score,
        skip_inference_below_threshold=args.skip_inference_below_threshold,
    )

    stop_requested = False

    def stop(_signum, _frame) -> None:  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nStopping.")
        return 0
    except Exception as exc:
        print(f"Audio stream error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if stop_requested:
            print("Stopping BLE server...", flush=True)
        usb_doa_reader.stop()
        ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
