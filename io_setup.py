from __future__ import annotations

import io
import sys


def setup_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 safe on Jetson/Linux terminals."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )

    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
