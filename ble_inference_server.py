from __future__ import annotations

import json
import sys

import realtime_inference_ble as ble


class AppInferenceCharacteristic(ble.InferenceCharacteristic):
    """앱이 바로 읽을 수 있는 단일 JSON BLE notify characteristic."""

    def _notify_latest(self) -> bool:
        if not self.notifying:
            return False

        self.sequence += 1
        payload_bytes = self.latest_payload.encode("utf-8")
        if len(payload_bytes) > self.chunk_bytes:
            print(
                f"warning: BLE JSON is {len(payload_bytes)} bytes; "
                f"larger than --ble-chunk-bytes={self.chunk_bytes}. "
                "If the app does not receive packets, increase Android MTU or omit --full-packet.",
                file=sys.stderr,
                flush=True,
            )

        self.PropertiesChanged(
            ble.GATT_CHRC_IFACE,
            ble.dbus.Dictionary({"Value": ble.byte_array(payload_bytes)}, signature="sv"),
            ble.dbus.Array([], signature="s"),
        )
        print(
            f"sent EdgeAudioRecognition notification seq={self.sequence} "
            f"bytes={len(payload_bytes)}",
            flush=True,
        )
        return False


class AppBleInferenceServer(ble.BleInferenceServer):
    """dict 데이터를 JSON으로 변환해 BLE로 publish."""

    def publish(self, data: dict) -> None:
        if self.characteristic is None:
            return
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        self.characteristic.notify_text(payload)


def install_app_compatible_ble_characteristic() -> None:
    ble.InferenceCharacteristic = AppInferenceCharacteristic
