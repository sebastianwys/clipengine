# ClipEngine

A local engine that analyzes a library of video clips, learns each clip's motion and color signature, and recommends which clips cut well together for short vertical edits. Everything runs on your machine: no cloud APIs, no telemetry, and the web UI binds to 127.0.0.1 only.

Day-to-day use and the editor handoff (Final Cut Pro, DaVinci Resolve) live in GUIDE.md. This file covers how it works.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export CLIPENGINE_MEDIA_ROOT="/path/to/footage"
```

The footage tree is `<root>/<camera profile>/<location>/clip.mp4`. Profile folders named in `config.py` are treated as log footage and color-normalized before analysis. Unknown folders fall back to automatic flat detection. OpenCV's wheel bundles its own decoder, so nothing is installed globally.

## Quick start

```bash
.venv/bin/python -m clipengine scan                 # catalog the tree, stat-only
.venv/bin/python -m clipengine analyze              # extract features, ~10 s per clip
.venv/bin/python -m clipengine status
.venv/bin/python -m clipengine match <clip> --mode=momentum
.venv/bin/python -m clipengine sequence --seed=<clip> --length=8
.venv/bin/python -m clipengine ui                   # http://127.0.0.1:8763
```

After tuning thresholds in `config.py`, `relabel` recomputes motion classes from stored vectors in seconds, with no video decoding.

## The cut model

A clip is not one thing, it is a start state and an end state. The unit being scored is always a cut: the end of clip A against the start of clip B. Each state stores:

- motion: mean optical-flow vector, energy, radial (push/pull), jitter, all in frame-widths per second so every camera and resolution is comparable
- a motion class: static, pan_left/right, tilt_up/down, whip_*, push_in, pull_out, handheld, drift, move_diagonal. Directional labels require steadiness (translation at least 0.75x jitter): a walking shot drifting rightward reads handheld, not pan_right, because an editor cannot cut on that drift
- color: luma, contrast, saturation, warmth (lab b*), tint (lab a*), and a 12-bin hue histogram weighted by saturation and value

Four scoring modes blend those components with different weights and a physical gate:

| mode     | idea                              | gate                       |
|----------|-----------------------------------|----------------------------|
| momentum | carry motion through the cut      | both sides must be moving  |
| whip     | blur-to-blur invisible cut        | both sides near whip speed |
| calm     | still-to-still, color led         | both sides near static     |
| contrast | deliberate vibe flip              | none (color terms invert)  |

Log profiles get a contrast stretch and saturation boost before color statistics are taken. Without that, every log clip reads as the same flat gray.

## Architecture

```
scan    catalog.py   stat-only walk -> sqlite catalog (never opens files)
analyze analysis.py  decode windows -> optical flow + color -> features
                     probe.py reads mp4 boxes, reader.py samples frames
match   matching.py  vectorized scoring of one clip against the library
chain   sequence.py  beam search over the cut-score matrix
export  fcpxml.py    frame-accurate timeline for fcp and resolve
ui      web/         localhost server, byte-range streaming, static page
```

## Data structures, and why each one

- sqlite catalog: two tables. `clips` is the inventory (path, profile, location, size, mtime, availability). `features` holds one row per analyzed clip. The join condition `version = ? AND content_key = clip.content_key` is the freshness rule: bump `FEATURE_VERSION` or touch a file on disk and the clip re-enters the analysis queue.
- content keys `"{size}-{mtime_ns}"`: cache invalidation without hashing, because hashing a cloud-evicted file would download it.
- fixed-layout float32 vector (50 slots): `features.py` names every slot, the catalog stores raw bytes. numpy `frombuffer` turns the whole library into one (n, 50) matrix, so scoring one clip against hundreds is a handful of vectorized operations, not a python loop.
- hue histogram (12 bins): a palette signature. Bins are weighted by saturation times value so gray sky and shadow cannot fake a color. Similarity is histogram intersection.
- energy profile (8 samples): a small time series of motion energy across the clip interior, drawn as a sparkline in the UI. Separates action from ambience at a glance.
- score matrix (n x n float32): entry [i, j] is the quality of the cut i -> j under one mode. The diagonal is -1 so a clip cannot follow itself. This is the weighted directed graph of all possible edits.
- beam search: greedy chaining walks into dead ends and exhaustive search is factorial. The beam keeps the 12 best partial chains alive each step. The test suite has a greedy-trap matrix that proves the difference.
- byte-range streaming: browsers scrub `<video>` with `Range: bytes=a-b` headers. The server answers 206 with exactly those bytes. Media is looked up by catalog id, never by path, and an evicted file is refused (409) rather than silently downloaded.

## Cloud-synced libraries

`scan` is stat-only: a file with `st_blocks` of 0 is a dataless stub and is cataloged as evicted. `analyze` and the UI only touch materialized clips. To analyze more, download the folders locally and rerun `scan` then `analyze`. Files over `MAX_ANALYZE_BYTES` are cataloged but skipped.

## Tuning

```bash
.venv/bin/python -m clipengine audit      # how the thresholds carve your footage
.venv/bin/python -m clipengine relabel    # recompute classes after editing config.py
```

| constant | value | meaning |
|----------|-------|---------|
| STATIC_MAX | 0.02 | below this a window is locked-off |
| PAN_MIN | 0.06 | deliberate directional move |
| WHIP_MIN | 0.60 | transition-grade speed, measured not physical |
| STEADY_RATIO | 0.75 | translation must be this fraction of jitter to count as directional |

Mode weights and gates are read at match time, so editing them needs no recompute. Changing the analysis itself (window length, flow parameters, `ANALYSIS_WIDTH`) means bumping `FEATURE_VERSION` and re-running `analyze`. Whip thresholds live in the measurement domain because optical flow underestimates very fast motion. `ANALYSIS_WIDTH` is part of the calibration: dense flow is more stable on downscaled frames, and 240 and 320 agree on real footage while 480 undermeasures.

## Exports

Every sequence export writes a json cut plan, an m3u8 playlist for a rough preview, and an fcpxml timeline that imports into Final Cut Pro and DaVinci Resolve. Times are frame-accurate rationals (2.002 s at 23.976 fps is exactly 48048/24000 s). Media is referenced in place by file url, nothing is copied or re-encoded.

## Tests

```bash
.venv/bin/python run_tests.py    # 85 tests, ~9 s
```

Analyzer tests run against synthetic clips whose motion is exact by construction: a texture rolled 4 px per frame is a 0.6 widths/sec pan, arithmetic, not opinion. Matching tests use hand-built vectors, the server tests exercise a live localhost server including range requests, and the integration test drives scan -> analyze -> match -> chain -> export end to end.

## License

MIT.
