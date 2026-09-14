# matching.py
# pairwise transition scoring. the unit being scored is always a cut:
# the end state of clip a against the start state of clip b. components
# are similarities in [0, 1]; a mode blends them with weights and a
# physical gate, so "momentum" cannot recommend a cut between two
# locked-off shots and "whip" cannot fire on slow footage.

import json
import sqlite3
from typing import Optional

import numpy as np

from clipengine import catalog, config
from clipengine.features import (END_HUE, HUE_BINS, INDEX, START_HUE,
                                 VECTOR_LEN, from_bytes)

_EPS = 1e-4


class Library:
    """in-memory matrix of every analyzed clip. row i of F belongs to
    ids[i] and meta[i]; row_of maps a clip id back to its row."""

    def __init__(self, ids: np.ndarray, F: np.ndarray, meta: list[dict]):
        self.ids = ids
        self.F = F
        self.meta = meta
        self.row_of = {int(cid): i for i, cid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.meta)


def load_library(conn: sqlite3.Connection,
                 country: Optional[str] = None) -> Library:
    """pull every fresh feature vector out of the catalog."""
    rows = catalog.analyzed(conn, country)
    ids, vecs, meta = [], [], []
    for r in rows:
        try:
            vec = from_bytes(r["vector"])
        except (ValueError, TypeError):
            continue
        summary = json.loads(r["summary"] or "{}")
        ids.append(r["id"])
        vecs.append(vec)
        meta.append({"id": r["id"], "name": r["name"],
                     "country": r["country"], "profile": r["profile"],
                     "path": r["path"],
                     "duration_s": summary.get("duration_s"),
                     "fps": summary.get("fps"),
                     "width": summary.get("width"),
                     "height": summary.get("height"),
                     "start_class": summary.get("start_class"),
                     "end_class": summary.get("end_class")})
    if not ids:
        return Library(np.zeros(0, np.int64),
                       np.zeros((0, VECTOR_LEN), np.float32), [])
    return Library(np.asarray(ids, np.int64), np.stack(vecs), meta)


def _col(F: np.ndarray, name: str) -> np.ndarray:
    return F[:, INDEX[name]]


def components(lib: Library, a_row: int) -> dict:
    """similarity components between clip a's end state and every clip's
    start state, vectorized over the whole library. arrays are (n,)."""
    F = lib.F
    a = F[a_row]
    ax, ay = a[INDEX["end_flow_x"]], a[INDEX["end_flow_y"]]
    ae = a[INDEX["end_energy"]]
    bx, by = _col(F, "start_flow_x"), _col(F, "start_flow_y")
    be = _col(F, "start_energy")

    # motion direction: cosine of the two flow vectors mapped to [0, 1].
    # when either side is basically still, direction is meaningless, so
    # the component goes neutral instead of rewarding noise alignment.
    dot = ax * bx + ay * by
    norm = np.hypot(ax, ay) * np.hypot(bx, by) + _EPS
    motion = (dot / norm + 1.0) / 2.0
    still = (ae < config.STATIC_MAX) | (be < config.STATIC_MAX)
    motion = np.where(still, 0.5, motion)

    # energy: speed ratio through the cut, 1.0 when both sides move alike
    energy = np.exp(-np.abs(np.log((ae + _EPS) / (be + _EPS))))

    # color: lab-ish distance on (luma, tint, warmth) plus palette overlap
    dl = a[INDEX["end_luma"]] - _col(F, "start_luma")
    da = a[INDEX["end_tint"]] - _col(F, "start_tint")
    db = a[INDEX["end_warmth"]] - _col(F, "start_warmth")
    de_sim = np.exp(-np.sqrt(dl * dl + da * da + db * db) / 40.0)
    hue_overlap = np.minimum(a[END_HUE][None, :], F[:, START_HUE]).sum(axis=1)
    color = 0.55 * de_sim + 0.45 * hue_overlap

    luma = 1.0 - np.abs(dl) / 255.0

    # physical gates per mode, each in [0, 1]
    gates = {
        "momentum": np.clip(np.minimum(ae, be) / config.PAN_MIN, 0.0, 1.0),
        # 0.8x keeps the whip gate demanding near-whip speed on both
        # sides even after WHIP_MIN was lowered to the library's tail
        "whip": np.clip(np.minimum(ae, be) / (0.8 * config.WHIP_MIN),
                        0.0, 1.0) ** 2,
        "calm": np.clip(1.0 - np.maximum(ae, be) / config.PAN_MIN, 0.0, 1.0),
    }
    gates["contrast"] = np.ones_like(motion)

    return {"motion": motion, "energy": energy, "color": color,
            "luma": luma, "gates": gates}


def score_against(lib: Library, a_row: int,
                  mode: str = config.DEFAULT_MODE
                  ) -> tuple[np.ndarray, dict]:
    """score clip a's cut into every other clip under one mode."""
    if mode not in config.SCORING_MODES:
        raise ValueError(f"unknown mode: {mode}")
    w = config.SCORING_MODES[mode]
    c = components(lib, a_row)
    # contrast mode rewards a deliberate vibe flip, so color and
    # brightness similarities invert while energy still has to carry
    color = 1.0 - c["color"] if mode == "contrast" else c["color"]
    luma = 1.0 - c["luma"] if mode == "contrast" else c["luma"]
    raw = (w["motion"] * c["motion"] + w["energy"] * c["energy"]
           + w["color"] * color + w["luma"] * luma)
    scores = raw * c["gates"][mode]
    scores[a_row] = -1.0
    return scores.astype(np.float32), c


def rank(lib: Library, clip_id: int, mode: str = config.DEFAULT_MODE,
         n: int = 10, country: str = "any") -> list[dict]:
    """top n candidate cuts out of clip_id, with score breakdowns."""
    if clip_id not in lib.row_of:
        raise KeyError(f"clip {clip_id} has no features yet")
    a_row = lib.row_of[clip_id]
    scores, c = score_against(lib, a_row, mode)
    a_country = lib.meta[a_row]["country"]
    out = []
    for i in np.argsort(scores)[::-1]:
        i = int(i)
        if len(out) >= n or scores[i] <= 0:
            break
        m = lib.meta[i]
        if country == "same" and m["country"] != a_country:
            continue
        if country == "different" and m["country"] == a_country:
            continue
        out.append({**m,
                    "score": round(float(scores[i]), 4),
                    "breakdown": {
                        "motion": round(float(c["motion"][i]), 3),
                        "energy": round(float(c["energy"][i]), 3),
                        "color": round(float(c["color"][i]), 3),
                        "luma": round(float(c["luma"][i]), 3),
                        "gate": round(float(c["gates"][mode][i]), 3)}})
    return out


def full_matrix(lib: Library, mode: str = config.DEFAULT_MODE) -> np.ndarray:
    """dense n x n cut-score matrix; entry [i, j] scores the cut i -> j.
    the diagonal is -1 so no clip can follow itself."""
    n = len(lib)
    M = np.full((n, n), -1.0, dtype=np.float32)
    for i in range(n):
        M[i], _ = score_against(lib, i, mode)
    return M
