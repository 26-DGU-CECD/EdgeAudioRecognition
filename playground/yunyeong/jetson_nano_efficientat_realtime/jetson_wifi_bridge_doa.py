#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import os
import sys
import time
import threading

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import jetson_wifi_bridge as bridge

DOA_ENABLE = os.environ.get("DOA_ENABLE", "1") != "0"
DOA_NORTH_OFFSET = float(os.environ.get("DOA_NORTH_OFFSET", "0"))

_doa_lock = threading.Lock()
_doa_started = False
_doa_angle = None
_doa_direction = None


def angle_to_cardinal(angle, north_offset=0.0):
    corrected = (float(angle) - float(north_offset)) % 360.0

    if corrected < 45 or corrected >= 315:
        return "북"
    elif corrected < 135:
        return "동"
    elif corrected < 225:
        return "남"
    else:
        return "서"


def doa_loop():
    global _doa_angle, _doa_direction

    try:
        import usb.core
        from tuning import Tuning

        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr)
            return

        tuning = Tuning(dev)
        print("[DOA] ReSpeaker DOA reader enabled.", file=sys.stderr)

        while True:
            try:
                angle = int(float(tuning.direction)) % 360
                direction = angle_to_cardinal(angle, DOA_NORTH_OFFSET)

                with _doa_lock:
                    _doa_angle = angle
                    _doa_direction = direction

            except Exception as e:
                print("[DOA] read error:", repr(e), file=sys.stderr)

            time.sleep(0.2)

    except Exception as e:
        print("[DOA] disabled:", repr(e), file=sys.stderr)


def ensure_doa_thread():
    global _doa_started

    if not DOA_ENABLE:
        return

    if _doa_started:
        return

    _doa_started = True
    t = threading.Thread(target=doa_loop)
    t.daemon = True
    t.start()


_orig_parse_line = bridge.parse_line


def parse_line_with_doa(line):
    parsed = _orig_parse_line(line)

    if parsed is None:
        return None

    ensure_doa_thread()

    with _doa_lock:
        angle = _doa_angle
        direction = _doa_direction

    if angle is None or direction is None:
        return parsed

    tag = "%s %d°" % (direction, angle)

    parsed["direction"] = direction
    parsed["angle"] = angle
    parsed["label"] = "%s [%s]" % (parsed["label"], tag)
    parsed["raw"] = "%s | DOA %s" % (parsed.get("raw", line.strip()), tag)

    if parsed.get("items"):
        parsed["items"][0]["label"] = "%s [%s]" % (parsed["items"][0]["label"], tag)

    return parsed


bridge.parse_line = parse_line_with_doa

if __name__ == "__main__":
    bridge.main()
