# fcpxml.py
# timeline handoff to real editors. final cut pro imports fcpxml
# natively and davinci resolve imports it via file > import timeline,
# so one exporter serves both. the tricky part is time: fcpxml times
# are rational numbers that must land exactly on frame boundaries, so
# every duration is expressed as frames * frame_duration, never as a
# float of seconds.

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

from clipengine import config

# common video rates as (seconds-per-frame numerator, denominator).
# 23.976 is really 24000/1001 fps, so one frame lasts 1001/24000 s.
_RATES = [
    (23.976, 1001, 24000),
    (24.0, 100, 2400),
    (25.0, 100, 2500),
    (29.97, 1001, 30000),
    (30.0, 100, 3000),
    (50.0, 100, 5000),
    (59.94, 1001, 60000),
    (60.0, 100, 6000),
    (119.88, 1001, 120000),
]


def frame_duration(fps: float) -> tuple[int, int]:
    """nearest standard rate as a rational seconds-per-frame."""
    if not fps or fps <= 0:
        fps = 24.0
    best = min(_RATES, key=lambda r: abs(r[0] - fps))
    if abs(best[0] - fps) <= 0.05:
        return best[1], best[2]
    whole = max(1, round(fps))
    return 100, whole * 100


def _rational(frames: int, num: int, den: int) -> str:
    return f"{frames * num}/{den}s"


def export_fcpxml(meta_rows: list[dict], mode: str,
                  out_dir: Optional[Path] = None,
                  stamp: Optional[str] = None) -> Path:
    """write a timeline of the given clips in order. each clip appears
    full length; trimming stays in the editor where it belongs. media
    is referenced in place by file url, never copied."""
    if out_dir is None:
        out_dir = config.EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")

    formats: dict[tuple, str] = {}
    def format_id(fps: float, width: int, height: int) -> str:
        num, den = frame_duration(fps)
        key = (num, den, width, height)
        if key not in formats:
            fid = f"r{len(formats) + 1}"
            formats[key] = fid
            attrs = {"id": fid, "frameDuration": f"{num}/{den}s"}
            if width and height:
                attrs["width"] = str(width)
                attrs["height"] = str(height)
            ET.SubElement(resources, "format", attrs)
        return formats[key]

    clips = []
    for i, m in enumerate(meta_rows):
        fps = float(m.get("fps") or 24.0)
        num, den = frame_duration(fps)
        fid = format_id(fps, int(m.get("width") or 0),
                        int(m.get("height") or 0))
        frames = max(1, round(float(m.get("duration_s") or 1.0) * den / num))
        duration = _rational(frames, num, den)
        aid = f"a{i + 1}"
        asset = ET.SubElement(
            resources, "asset",
            {"id": aid, "name": str(m["name"]), "start": "0s",
             "duration": duration, "hasVideo": "1", "hasAudio": "1",
             "format": fid})
        ET.SubElement(asset, "media-rep",
                      {"kind": "original-media",
                       "src": Path(m["path"]).as_uri()})
        clips.append((aid, str(m["name"]), duration))

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="ClipEngine")
    project = ET.SubElement(event, "project",
                            name=f"clipengine {mode} {stamp}")
    first_format = next(iter(formats.values())) if formats else "r1"
    seq = ET.SubElement(project, "sequence", format=first_format)
    spine = ET.SubElement(seq, "spine")
    for aid, name, duration in clips:
        # no offsets: spine children without offsets lay out end to end,
        # which both fcp and resolve honor
        ET.SubElement(spine, "asset-clip",
                      {"ref": aid, "name": name, "duration": duration})

    body = ET.tostring(root, encoding="unicode")
    text = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE fcpxml>\n\n" + body + "\n")
    path = out_dir / f"sequence_{stamp}.fcpxml"
    path.write_text(text)
    return path
