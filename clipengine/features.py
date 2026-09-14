# features.py
# fixed-layout feature vector. every analyzed clip becomes one float32
# array whose slots are named here. the catalog stores the raw bytes;
# this module is the single source of truth for what each slot means.

import numpy as np

HUE_BINS = 12

_SCALARS = [
    # motion state of the first window (frame-widths per second)
    "start_flow_x", "start_flow_y", "start_energy", "start_radial", "start_jitter",
    # motion state of the last window
    "end_flow_x", "end_flow_y", "end_energy", "end_radial", "end_jitter",
    # color state of the first window (8-bit scales)
    "start_luma", "start_contrast", "start_sat", "start_warmth", "start_tint",
    # color state of the last window
    "end_luma", "end_contrast", "end_sat", "end_warmth", "end_tint",
    # whole-clip context
    "global_luma", "global_sat", "global_warmth",
    "tempo_mean", "tempo_var", "duration_s",
]

FIELDS = (_SCALARS
          + [f"start_hue_{i}" for i in range(HUE_BINS)]
          + [f"end_hue_{i}" for i in range(HUE_BINS)])
INDEX = {name: i for i, name in enumerate(FIELDS)}
VECTOR_LEN = len(FIELDS)

START_HUE = slice(INDEX["start_hue_0"], INDEX["start_hue_0"] + HUE_BINS)
END_HUE = slice(INDEX["end_hue_0"], INDEX["end_hue_0"] + HUE_BINS)


def pack(scalars: dict, start_hue: np.ndarray, end_hue: np.ndarray) -> np.ndarray:
    """assemble the canonical vector. missing scalar names raise so a
    drifting analyzer fails loudly instead of writing garbage."""
    vec = np.zeros(VECTOR_LEN, dtype=np.float32)
    for name in _SCALARS:
        vec[INDEX[name]] = float(scalars[name])
    vec[START_HUE] = np.asarray(start_hue, dtype=np.float32)
    vec[END_HUE] = np.asarray(end_hue, dtype=np.float32)
    return vec


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.shape[0] != VECTOR_LEN:
        raise ValueError(f"expected {VECTOR_LEN} floats, got {vec.shape[0]}")
    return vec.copy()


def get(vec: np.ndarray, name: str) -> float:
    return float(vec[INDEX[name]])
