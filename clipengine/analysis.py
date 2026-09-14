# analysis.py
# turns one clip into features: optical-flow motion for the first and
# last windows, log-aware color statistics, a mid-clip energy profile,
# and start/mid/end thumbnails.

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from clipengine import config, features, probe, reader

logger = logging.getLogger(__name__)

# five pyramid levels so whip-fast displacements are still trackable at
# the coarsest scale; three levels measurably undershot fast pans
_FARNEBACK = dict(pyr_scale=0.5, levels=5, winsize=21, iterations=3,
                  poly_n=5, poly_sigma=1.2, flags=0)


class AnalysisError(Exception):
    """raised when a clip cannot produce a full feature set."""


@dataclass
class WindowMotion:
    flow_x: float
    flow_y: float
    energy: float
    radial: float
    jitter: float
    label: str = ""


# -- motion --------------------------------------------------------------

def _radial_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """unit vectors pointing away from frame center, for push/pull."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - (w - 1) / 2.0, ys - (h - 1) / 2.0
    r = np.sqrt(dx * dx + dy * dy)
    r[r < 1.0] = 1.0
    return dx / r, dy / r


def window_motion(frames_bgr: list[np.ndarray], fps: float) -> WindowMotion:
    """aggregate dense optical flow over a window into one motion state.
    all values are in frame-widths per second, resolution independent."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    h, w = grays[0].shape
    ux, uy = _radial_grid(h, w)
    scale = fps / w
    txs, tys, mags, rads = [], [], [], []
    for prev, curr in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **_FARNEBACK)
        fx, fy = flow[..., 0], flow[..., 1]
        txs.append(float(fx.mean()) * scale)
        tys.append(float(fy.mean()) * scale)
        mags.append(float(np.sqrt(fx * fx + fy * fy).mean()) * scale)
        rads.append(float((fx * ux + fy * uy).mean()) * scale)
    m = WindowMotion(
        flow_x=float(np.mean(txs)),
        flow_y=float(np.mean(tys)),
        energy=float(np.mean(mags)),
        radial=float(np.mean(rads)),
        jitter=float(np.hypot(np.std(txs), np.std(tys))))
    m.label = classify_motion(m)
    return m


def classify_motion(m: WindowMotion) -> str:
    """label a window's camera move.

    flow measures apparent content motion in image coordinates (x right,
    y down), so labels invert the sign to name the camera's direction:
    a camera panning right makes content flow left (flow_x < 0), a tilt
    up makes content flow down (flow_y > 0). push_in is expanding radial
    flow: walking or zooming forward.
    """
    t = float(np.hypot(m.flow_x, m.flow_y))
    steady = t >= config.STEADY_RATIO * m.jitter
    if m.energy < config.STATIC_MAX:
        return "static"
    if (abs(m.radial) >= config.PUSH_MIN
            and abs(m.radial) >= config.RADIAL_DOMINANCE * t):
        return "push_in" if m.radial > 0 else "pull_out"
    if t >= config.WHIP_MIN and steady:
        return "whip_" + _dir4(m.flow_x, m.flow_y)
    if t >= config.PAN_MIN and steady:
        d = _dir4(m.flow_x, m.flow_y)
        if d in ("left", "right"):
            return "pan_" + d
        if d in ("up", "down"):
            return "tilt_" + d
        return "move_diagonal"
    if m.jitter > max(1.5 * t, config.STATIC_MAX):
        return "handheld"
    return "drift"


def _dir4(fx: float, fy: float) -> str:
    """camera-direction name from a content-flow vector."""
    if abs(fx) >= config.AXIS_DOMINANCE * abs(fy):
        return "right" if fx < 0 else "left"
    if abs(fy) >= config.AXIS_DOMINANCE * abs(fx):
        return "up" if fy > 0 else "down"
    return "diagonal"


# -- color ----------------------------------------------------------------

def _pixels(frames_bgr: list[np.ndarray], cap: int = 6) -> np.ndarray:
    """stack a spread of frames into one (1, n, 3) image for stats."""
    if len(frames_bgr) > cap:
        idx = np.linspace(0, len(frames_bgr) - 1, cap).round().astype(int)
        frames_bgr = [frames_bgr[i] for i in idx]
    px = np.concatenate([f.reshape(-1, 3) for f in frames_bgr])
    return px.reshape(1, -1, 3)


def normalize_flat(img_bgr: np.ndarray) -> np.ndarray:
    """expand a log image toward display contrast: stretch the l channel
    between its 5th and 95th percentiles, then boost saturation. crude
    next to a real lut, but consistent across clips, and consistency is
    what matching needs."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[..., 0]
    p5, p95 = np.percentile(l_chan, (5, 95))
    if p95 - p5 > 1.0:
        lab[..., 0] = np.clip((l_chan - p5) * (235.0 - 16.0) / (p95 - p5) + 16.0,
                              0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.6, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def color_stats(frames_bgr: list[np.ndarray], force_log: bool) -> dict:
    """color state of a window. log footage is normalized before stats so
    flat capture does not make every clip read as the same gray vibe.
    hue histogram weights each pixel by saturation and value, so gray sky
    and shadows do not pollute the palette signature."""
    img = _pixels(frames_bgr)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    l_chan = lab[..., 0].ravel()
    p10, p90 = np.percentile(l_chan, (10, 90))
    sat_raw = float(hsv[..., 1].mean())
    flat = bool(force_log or (sat_raw < config.FLAT_SAT_MAX
                              and (p90 - p10) < config.FLAT_CONTRAST_MAX))
    if flat:
        img = normalize_flat(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        l_chan = lab[..., 0].ravel()
        p10, p90 = np.percentile(l_chan, (10, 90))
    h = hsv[..., 0].ravel()
    weights = (hsv[..., 1].ravel() / 255.0) * (hsv[..., 2].ravel() / 255.0)
    hist, _ = np.histogram(h, bins=features.HUE_BINS, range=(0.0, 180.0),
                           weights=weights)
    total = float(hist.sum())
    if total > 1e-6:
        hue = (hist / total).astype(np.float32)
    else:
        hue = np.full(features.HUE_BINS, 1.0 / features.HUE_BINS, np.float32)
    return {"luma": float(l_chan.mean()),
            "contrast": float(p90 - p10),
            "sat": float(hsv[..., 1].mean()),
            "warmth": float(lab[..., 2].mean() - 128.0),
            "tint": float(lab[..., 1].mean() - 128.0),
            "hue": hue,
            "flat": flat}


# -- energy profile ---------------------------------------------------------

def energy_profile(path, duration_s: float,
                   samples: int = config.ENERGY_SAMPLES) -> np.ndarray:
    """flow-magnitude probes across the clip interior, widths/sec. gives
    the tempo curve used to distinguish action clips from ambience."""
    if duration_s <= 0:
        return np.zeros(0, dtype=np.float32)
    vals = []
    for frac in np.linspace(0.05, 0.95, samples):
        pair = reader.read_pair(path, float(frac * duration_s))
        if pair is None:
            continue
        f1, f2, fps = pair
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **_FARNEBACK)
        mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        vals.append(mag * fps / g1.shape[1])
    return np.asarray(vals, dtype=np.float32)


# -- thumbnails --------------------------------------------------------------

def make_thumb(frame_bgr: np.ndarray, flat: bool) -> bytes:
    """jpeg thumbnail, display-normalized when the source is log."""
    img = normalize_flat(frame_bgr) if flat else frame_bgr
    h, w = img.shape[:2]
    if w > config.THUMB_WIDTH:
        img = cv2.resize(img, (config.THUMB_WIDTH, round(h * config.THUMB_WIDTH / w)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, config.THUMB_QUALITY])
    if not ok:
        raise AnalysisError("jpeg encode failed")
    return bytes(buf)


def motion_from_vector(vec: np.ndarray, side: str) -> WindowMotion:
    """rebuild a window's motion state from a stored feature vector.
    labels are derived, not stored in the vector, so threshold changes
    can relabel the whole catalog without decoding any video."""
    get = features.get
    m = WindowMotion(
        flow_x=get(vec, f"{side}_flow_x"),
        flow_y=get(vec, f"{side}_flow_y"),
        energy=get(vec, f"{side}_energy"),
        radial=get(vec, f"{side}_radial"),
        jitter=get(vec, f"{side}_jitter"))
    m.label = classify_motion(m)
    return m


# -- orchestration -----------------------------------------------------------

@dataclass
class AnalysisResult:
    vector: np.ndarray
    summary: dict
    thumbs: dict  # position -> jpeg bytes


def _window_summary(m: WindowMotion, c: dict) -> dict:
    return {"class": m.label,
            "flow": [round(m.flow_x, 4), round(m.flow_y, 4)],
            "energy": round(m.energy, 4),
            "radial": round(m.radial, 4),
            "jitter": round(m.jitter, 4),
            "luma": round(c["luma"], 1),
            "contrast": round(c["contrast"], 1),
            "sat": round(c["sat"], 1),
            "warmth": round(c["warmth"], 1),
            "tint": round(c["tint"], 1)}


def analyze_clip(path, is_log: bool) -> AnalysisResult:
    """full feature extraction for one materialized clip."""
    meta = probe.probe(path)
    duration = float(meta.get("duration_s") or 0.0)

    start_frames, fps = reader.read_window(path, 0.0)
    if duration <= 0:
        duration = len(start_frames) / fps
    end_start = max(0.0, duration - config.WINDOW_SECONDS - 0.05)
    end_frames, _ = reader.read_window(path, end_start)

    m_start = window_motion(start_frames, fps)
    m_end = window_motion(end_frames, fps)
    c_start = color_stats(start_frames, is_log)
    c_end = color_stats(end_frames, is_log)

    mid_frame = reader.read_frame(path, duration / 2.0, config.ANALYSIS_WIDTH)
    global_sample = start_frames[::4] + end_frames[::4]
    if mid_frame is not None:
        global_sample.append(mid_frame)
    c_global = color_stats(global_sample, is_log)

    profile = energy_profile(path, duration)
    tempo_mean = (float(profile.mean()) if profile.size
                  else (m_start.energy + m_end.energy) / 2.0)
    tempo_var = float(profile.var()) if profile.size else 0.0

    scalars = {
        "start_flow_x": m_start.flow_x, "start_flow_y": m_start.flow_y,
        "start_energy": m_start.energy, "start_radial": m_start.radial,
        "start_jitter": m_start.jitter,
        "end_flow_x": m_end.flow_x, "end_flow_y": m_end.flow_y,
        "end_energy": m_end.energy, "end_radial": m_end.radial,
        "end_jitter": m_end.jitter,
        "start_luma": c_start["luma"], "start_contrast": c_start["contrast"],
        "start_sat": c_start["sat"], "start_warmth": c_start["warmth"],
        "start_tint": c_start["tint"],
        "end_luma": c_end["luma"], "end_contrast": c_end["contrast"],
        "end_sat": c_end["sat"], "end_warmth": c_end["warmth"],
        "end_tint": c_end["tint"],
        "global_luma": c_global["luma"], "global_sat": c_global["sat"],
        "global_warmth": c_global["warmth"],
        "tempo_mean": tempo_mean, "tempo_var": tempo_var,
        "duration_s": duration,
    }
    vector = features.pack(scalars, c_start["hue"], c_end["hue"])

    flat = bool(c_start["flat"] or c_end["flat"])
    summary = {
        "duration_s": round(duration, 3),
        "fps": round(fps, 3),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
        "codec": meta.get("codec", ""),
        "created": meta.get("created"),
        "flat": flat,
        "start_class": m_start.label,
        "end_class": m_end.label,
        "start": _window_summary(m_start, c_start),
        "end": _window_summary(m_end, c_end),
        "tempo_mean": round(tempo_mean, 4),
        "tempo_var": round(tempo_var, 5),
        "energy_profile": [round(float(x), 4) for x in profile],
    }

    thumbs = {"start": make_thumb(start_frames[len(start_frames) // 2], flat),
              "end": make_thumb(end_frames[len(end_frames) // 2], flat)}
    thumbs["mid"] = (make_thumb(mid_frame, flat) if mid_frame is not None
                     else thumbs["start"])
    return AnalysisResult(vector=vector, summary=summary, thumbs=thumbs)
