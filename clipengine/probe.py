# probe.py
# clip metadata without external tools. an mp4/mov file is a tree of
# length-prefixed "boxes"; we walk the top level, find moov/mvhd, and
# read timescale, duration, and creation time directly. opencv fills in
# fps and geometry through its bundled ffmpeg decoder.

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

# quicktime epoch: seconds since 1904-01-01 utc
_QT_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
_CONTAINER_EXTS = {".mp4", ".mov"}


def _iter_boxes(f, end: int):
    """yield (type, payload_start, box_end) for consecutive boxes.
    handles 64-bit largesize (size == 1) and to-end boxes (size == 0)."""
    while True:
        pos = f.tell()
        if pos + 8 > end:
            return
        header = f.read(8)
        if len(header) < 8:
            return
        size, btype = struct.unpack(">I4s", header)
        payload_start = pos + 8
        if size == 1:
            large = f.read(8)
            if len(large) < 8:
                return
            size = struct.unpack(">Q", large)[0]
            payload_start = pos + 16
        elif size == 0:
            size = end - pos
        if size < 8 or pos + size > end:
            return
        yield btype, payload_start, pos + size
        f.seek(pos + size)


def parse_container(path) -> dict:
    """duration and creation time from the mvhd box. returns {} when the
    structure is absent or malformed (mxf, truncated files)."""
    result: dict = {}
    try:
        total = Path(path).stat().st_size
        with open(path, "rb") as f:
            for btype, start, bend in _iter_boxes(f, total):
                if btype != b"moov":
                    continue
                f.seek(start)
                for ctype, cstart, _cend in _iter_boxes(f, bend):
                    if ctype != b"mvhd":
                        continue
                    f.seek(cstart)
                    version = f.read(4)[0]
                    if version == 1:
                        ctime, _m, timescale, duration = struct.unpack(
                            ">QQIQ", f.read(28))
                    else:
                        ctime, _m, timescale, duration = struct.unpack(
                            ">IIII", f.read(16))
                    if timescale:
                        result["duration_s"] = duration / timescale
                    if ctime:
                        created = _QT_EPOCH + timedelta(seconds=ctime)
                        result["created"] = created.isoformat(timespec="seconds")
                    return result
                return result
    except (OSError, struct.error, IndexError):
        pass
    return result


def probe_cv2(path) -> dict:
    """fps, geometry, frame count, and codec tag from the decoder."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = ""
        if fourcc:
            codec = "".join(chr((fourcc >> 8 * i) & 0xFF)
                            for i in range(4)).strip("\x00 ")
        out = {"fps": fps if 1.0 <= fps <= 240.0 else 30.0,
               "frames": frames,
               "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
               "codec": codec}
        if frames > 0:
            out["duration_cv2"] = frames / out["fps"]
        return out
    finally:
        cap.release()


def probe(path) -> dict:
    """merged metadata. container duration wins when present; decoder
    frame math is the fallback."""
    meta = probe_cv2(path)
    if Path(path).suffix.lower() in _CONTAINER_EXTS:
        meta.update(parse_container(path))
    meta["duration_s"] = meta.get("duration_s") or meta.get("duration_cv2") or 0.0
    return meta
