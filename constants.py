from __future__ import annotations

import numpy as np

CARDINAL_SUFFIX = {
    "북": "북쪽",
    "동": "동쪽",
    "남": "남쪽",
    "서": "서쪽",
}

DANGER_LABELS = {
    "gunshot",
    "alarm_siren",
    "horn",
    "glass_shatter",
}

CAUTION_LABELS = {
    "construction",
    "water",
    "knock",
    "appliances",
    "baby_cry",
    "animal_cry",
}

RESPEAKER_USB_VENDOR_ID = 0x2886
RESPEAKER_USB_PRODUCT_ID = 0x0018
SPEED_OF_SOUND_M_S = 343.0
RAW_DOA_CHANNELS = (1, 2, 3, 4)
RAW_DOA_MIC_POSITIONS_M = np.asarray(
    [
        (-0.032, 0.000),
        (0.000, -0.032),
        (0.032, 0.000),
        (0.000, 0.032),
    ],
    dtype=np.float32,
)
