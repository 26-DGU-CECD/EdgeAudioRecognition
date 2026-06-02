#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys

import torch

from audio_buffer import AudioBuffer
from audio_doa_estimator import AudioDOAEstimator
from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from audio_stream_controller import AudioStreamController
from ble_inference_server import AppBleInferenceServer, install_app_compatible_ble_characteristic
from cli import parse_args
from db_threshold_gate import DbThresholdGate
from doa_reader import DOAReader
from doa_selector import DOASelector
from io_setup import setup_utf8_stdio
from microphone_module import MicrophoneModule
from model_runner import SoundRecognitionModel
from packet_builder import AppSoundPacketBuilder
from realtime_inference import CHUNK_SAMPLES, SAMPLE_RATE, find_respeaker_device, print_input_devices


def main() -> int:
    setup_utf8_stdio()
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = find_respeaker_device(args.device_index)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    sound_model = SoundRecognitionModel(args.efficientat_dir, device)
    try:
        sound_model.load()
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
        stream_channels=stream_channels,
        sample_rate=SAMPLE_RATE,
        min_db=args.audio_doa_min_db,
        window_ms=args.audio_doa_window_ms,
    )

    microphone = MicrophoneModule(
        device_index=device_index,
        sample_rate=SAMPLE_RATE,
        channels=stream_channels,
    )
    audio_queue = AudioQueue()
    audio_buffer = AudioBuffer(CHUNK_SAMPLES)
    level_meter = AudioLevelMeter()
    threshold_gate = DbThresholdGate(args.min_db)
    preprocessor = AudioPreprocessor(
        enhance_threshold_db=args.enhance_threshold_db,
        noise_reduction_db=args.noise_reduction_db,
        main_gain_db=args.main_gain_db,
        enhance_sharpness=args.enhance_sharpness,
    )
    doa_selector = DOASelector(
        doa_source=args.doa_source,
        usb_reader=usb_doa_reader,
        audio_estimator=audio_doa_estimator,
    )
    packet_builder = AppSoundPacketBuilder(
        north_offset=args.north_offset,
        db_offset=args.db_offset,
        full_packet=args.full_packet,
    )

    controller = AudioStreamController(
        device_index=device_index,
        device_info=device_info,
        stream_channels=stream_channels,
        channel_index=args.channel_index,
        debug=args.debug,
        min_score=args.min_score,
        min_db=args.min_db,
        ble_server=ble_server,
        microphone=microphone,
        audio_queue=audio_queue,
        audio_buffer=audio_buffer,
        level_meter=level_meter,
        threshold_gate=threshold_gate,
        preprocessor=preprocessor,
        model=sound_model,
        doa_selector=doa_selector,
        packet_builder=packet_builder,
        audio_doa_min_dbfs=audio_doa_estimator.min_dbfs,
        usb_doa_status=usb_doa_reader.status,
        audio_doa_status=audio_doa_estimator.status,
        doa_source=args.doa_source,
        north_offset=args.north_offset,
        db_offset=args.db_offset,
        skip_inference_below_threshold=True,
    )

    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        controller.process_loop()
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
