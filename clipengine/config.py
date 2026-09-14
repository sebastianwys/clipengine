# config.py
# central configuration: paths, camera profiles, analysis parameters,
# scoring modes. everything runs local to this machine: no network calls,
# no cloud apis, ui bound to loopback only.

import os
from pathlib import Path

# -- paths --------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# override with CLIPENGINE_MEDIA_ROOT
_default_root = Path.home() / "Documents" / "Media" / "The Footage"
MEDIA_ROOT = Path(os.environ.get("CLIPENGINE_MEDIA_ROOT", _default_root)).expanduser()
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "catalog.db"
THUMB_DIR = DATA_DIR / "thumbs"
EXPORT_DIR = BASE_DIR / "exports"

# -- media tree semantics ------------------------------------------------

# top-level folders under MEDIA_ROOT are camera profile folders, second
# level is country (the vibe tag). the profiles listed here shoot log,
# so color analysis must normalize before judging vibe. unknown folders
# fall back to automatic flat detection.
CAMERA_PROFILES = {
    "Sony SLOG-3": {"key": "sony_slog3", "log": True},
    "DJI DLOG-M": {"key": "dji_dlogm", "log": True},
    "Apple ProRes-Log": {"key": "apple_prores_log", "log": True},
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf"}

# cataloged but never analyzed: one oversize file must not stall a whole pass
MAX_ANALYZE_BYTES = 20 * 1024**3

# -- analysis parameters --------------------------------------------------

FEATURE_VERSION = 1      # bump to force re-analysis after algorithm changes
# 320 is the calibrated reference: motion thresholds below were tuned at
# this width, and dense flow behaves differently at other scales (the
# downscale itself stabilizes tracking on blurred, low-texture footage).
# changing this means bumping FEATURE_VERSION and re-analyzing.
ANALYSIS_WIDTH = 320
WINDOW_SECONDS = 1.2     # length of the start and end windows
MAX_WINDOW_FRAMES = 36   # cap decoded frames per window on high-fps sources
ENERGY_SAMPLES = 8       # mid-clip flow probes for the energy profile
THUMB_WIDTH = 480
THUMB_QUALITY = 85

# motion thresholds. units are frame-widths per second of apparent
# content motion, so values are resolution independent.
STATIC_MAX = 0.02        # below this the window is a locked-off shot
PAN_MIN = 0.06           # deliberate directional move
# whips live in the measurement domain, not the physical one: optical
# flow underestimates very fast motion (aliasing, motion blur), so this
# is calibrated against what farneback reports. 2026-07 library audit:
# window-energy p90 0.271, p99 0.641, so 0.85 sat above p99 and starved
# the class (2 of 438 windows). 0.60 is ~p98.5: whips stay rare but the
# genuinely fast tail qualifies. the steadiness rule still excludes
# chaotic shake. retune with `audit` + `relabel` as the library grows.
WHIP_MIN = 0.60
AXIS_DOMINANCE = 2.0     # |x| vs |y| ratio to call pan vs tilt
RADIAL_DOMINANCE = 1.4   # radial vs translation ratio to call push/pull
PUSH_MIN = 0.030         # radial expansion threshold
# a directional label must be steadier than it is shaky: translation has
# to reach this fraction of jitter, or the window is handheld. a walking
# shot drifting right is not a pan an editor can cut on.
STEADY_RATIO = 0.75

# flat / log detection on the 8-bit scale
FLAT_SAT_MAX = 60.0      # mean saturation below this suggests log capture
FLAT_CONTRAST_MAX = 90.0 # l-channel p90-p10 below this suggests log

# -- transition scoring modes ---------------------------------------------

# weights blend component similarities into one score in [0, 1]. gates
# (in matching.py) dampen a mode when the physics are wrong for it: a
# whip cut needs both sides moving fast, a calm cut needs both still.
SCORING_MODES = {
    "momentum": {"motion": 0.45, "energy": 0.20, "color": 0.25, "luma": 0.10},
    "whip":     {"motion": 0.55, "energy": 0.30, "color": 0.15, "luma": 0.00},
    "calm":     {"motion": 0.00, "energy": 0.20, "color": 0.55, "luma": 0.25},
    "contrast": {"motion": 0.00, "energy": 0.30, "color": 0.50, "luma": 0.20},
}
DEFAULT_MODE = "momentum"

# -- web ui ----------------------------------------------------------------

# hard-bound to loopback. this engine is private to this machine by
# design; never change this to 0.0.0.0.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8763


def create_directories() -> None:
    """create runtime directories if missing."""
    for d in (DATA_DIR, THUMB_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
