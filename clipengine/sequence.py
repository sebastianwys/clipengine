# sequence.py
# orders clips into a chain by maximizing summed cut scores over the
# transition graph. greedy would lock in early mistakes and exhaustive
# search is factorial, so a beam search keeps the best b partial chains
# alive at each step: the standard middle ground.

import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from clipengine import config


def build_chain(matrix: np.ndarray, seed_row: int, length: int,
                beam_width: int = 12,
                countries: Optional[list[str]] = None,
                country_mode: str = "any") -> tuple[list[int], list[float]]:
    """returns (rows, edge_scores): matrix row indices in play order and
    the score of each cut. country_mode 'same' keeps the seed's country,
    'travel' dampens consecutive same-country cuts to force variety."""
    n = matrix.shape[0]
    if not 0 <= seed_row < n:
        raise IndexError(f"seed row {seed_row} outside matrix of {n}")
    length = max(1, min(length, n))
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, (seed_row,))]

    for _ in range(length - 1):
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for total, path in beams:
            last = path[-1]
            visited = set(path)
            row = matrix[last]
            added = 0
            for j in np.argsort(row)[::-1]:
                j = int(j)
                if j in visited or row[j] <= 0:
                    continue
                edge = float(row[j])
                if countries is not None:
                    if (country_mode == "same"
                            and countries[j] != countries[seed_row]):
                        continue
                    if (country_mode == "travel"
                            and countries[j] == countries[last]):
                        edge *= 0.6
                candidates.append((total + edge, path + (j,)))
                added += 1
                if added >= beam_width:
                    break
        if not candidates:
            break
        beams = heapq.nlargest(beam_width, candidates, key=lambda b: b[0])

    _, best_path = max(beams, key=lambda b: b[0])
    edges = [float(matrix[a, b]) for a, b in zip(best_path, best_path[1:])]
    return list(best_path), edges


def export_chain(meta_rows: list[dict], edges: list[float], mode: str,
                 out_dir: Optional[Path] = None) -> tuple[Path, Path, Path]:
    """write the sequence three ways and return the paths: a json cut
    plan (machine readable), an m3u8 playlist (rough order preview in
    iina/vlc), and an fcpxml timeline (import into final cut pro or
    davinci resolve)."""
    from clipengine import fcpxml
    if out_dir is None:
        out_dir = config.EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = {"mode": mode,
            "created": stamp,
            "total_score": round(sum(edges), 4),
            "clips": [{"order": i,
                       "id": m["id"],
                       "name": m["name"],
                       "country": m["country"],
                       "path": m["path"],
                       "cut_score_from_prev": (round(edges[i - 1], 4)
                                               if i else None)}
                      for i, m in enumerate(meta_rows)]}
    json_path = out_dir / f"sequence_{stamp}.json"
    json_path.write_text(json.dumps(plan, indent=2))

    lines = ["#EXTM3U"]
    for m in meta_rows:
        lines.append(f"#EXTINF:{m.get('duration_s') or 0},{m['name']}")
        lines.append(str(m["path"]))
    m3u_path = out_dir / f"sequence_{stamp}.m3u8"
    m3u_path.write_text("\n".join(lines) + "\n")

    xml_path = fcpxml.export_fcpxml(meta_rows, mode, out_dir, stamp)
    return json_path, m3u_path, xml_path
