# ClipEngine User Guide

How to run the engine day to day, get sequences out of it, and land them in Final Cut Pro or DaVinci Resolve with log footage handled correctly. README.md covers how it works.

## 1. The cockpit

```bash
.venv/bin/python -m clipengine ui
```

Open http://127.0.0.1:8763. Safari decodes Sony and DJI HEVC and iPhone ProRes natively. Chrome shows thumbnails and scores but may refuse to play previews. The server binds to 127.0.0.1 only. Stop it with ctrl-c. If the port is taken, add `--port=8764`.

- grid: every analyzed clip. Hover a card to cycle its start, mid and end frames. Filters: location, motion class, name.
- detail panel: scrubbable preview, motion classes, color state, the energy sparkline.
- find matches: ranks the best cuts out of the selected clip under the current mode. Each row shows end-of-A beside start-of-B, the score, and four component bars (motion, energy, color, luma).
- sequence tray: add matches by hand, or pick a seed clip and hit auto-build to let beam search chain it. Export writes the three files in section 5.

## 2. The footage lifecycle

```
import footage -> scan (stat-only, seconds)
              -> analyze (only new or changed clips)
              -> browse / match / sequence in the ui
              -> export -> import into fcp or resolve
```

`scan` never opens files, so it is always safe to run. `analyze` only touches clips that are local and either new or changed. After downloading more footage from a cloud-synced library, run `scan` then `analyze`.

## 3. Choosing a mode

| mode     | the cut it builds                           | use it for |
|----------|---------------------------------------------|------------|
| momentum | motion direction carries across the cut     | walking or driving energy, travel montages, the default |
| whip     | blur-to-blur, cut hidden inside fast motion | high-energy transitions between locations |
| calm     | still-to-still, matched color               | intros, outros, ambience beds |
| contrast | deliberate vibe flip (color, brightness)    | section changes: place to place, day to night |

Location scope refines any mode: **same** keeps one place's story, **different** forces variety for montage pacing.

Whip mode needs whip footage. At the end of a shot, whip the camera hard in one direction. Start the next shot whipping the same direction. The engine finds and pairs them.

## 4. Reading a score

A score near 1.0 means the cut should feel invisible or intentional. The bars say why a candidate ranked where it did:

- motion: flow direction agreement across the cut, the big one
- energy: speed ratio, 1.0 means both sides move at the same pace
- color: palette plus warmth and luma continuity, inverted in contrast mode
- luma: brightness continuity

A high score with a weak color bar is still a usable cut. A high score with a weak motion bar in momentum mode is not. Trust motion first.

## 5. What export produces

Every export writes three files into `exports/`:

| file | what it is | open with |
|------|------------|-----------|
| `sequence_<stamp>.json`   | the cut plan: order, locations, per-cut scores | anything |
| `sequence_<stamp>.m3u8`   | playlist of the clips in order | IINA or VLC for a rough preview |
| `sequence_<stamp>.fcpxml` | a real timeline | Final Cut Pro or DaVinci Resolve |

The fcpxml references your originals in place by file url. Nothing is copied or re-encoded, and every clip must be local before the editor can play it.

## 6. Final Cut Pro

1. File > Import > XML and pick the `.fcpxml`. You get an event named ClipEngine with the clips laid end to end.
2. Log conversion: Apple Log is recognized automatically. For Sony clips, select them, open the Info inspector, and set Camera LUT to S-Log3/S-Gamut3.Cine. For DJI, apply a D-Log M LUT the same way.
3. Trim with the match in mind. The engine matched the end of each clip to the start of the next. Keep roughly the last second of the outgoing clip and the first second of the incoming clip, and trim the other ends.
4. Whip cuts: add a 2 to 4 frame overlap and, for more violence, a short speed ramp into the cut.
5. Vertical delivery: duplicate the project, Modify Project, set the format to 1080x1920, use Smart Conform as a starting point.

## 7. DaVinci Resolve

1. File > Import Timeline > Import AAF, EDL, XML, FCPXML and pick the `.fcpxml`. Leave "automatically import source clips" checked. If anything comes in offline, Relink Selected Clips and point it at your footage root.
2. Color-manage the log footage. Project-wide: Color Science DaVinci YRGB Color Managed. Manual: one group per camera profile with a Color Space Transform in the pre-clip stage, input S-Log3, D-Log M or Apple Log, output Rec.709 Gamma 2.4.
3. Deliver for vertical platforms at 1080x1920, H.264 High profile around 10 to 15 Mbps, AAC audio. The platforms re-encode anyway.

## 8. The local rule

A clip must be local to be played, analyzed or edited. The engine catalogs evicted stubs but refuses to touch their bytes. The editors are less polite: an evicted clip imports as offline media. Before an editing session, check the clips play in the UI preview, download anything missing, run `scan`, and re-export.

## 9. Troubleshooting

- previews will not play in the browser: use Safari.
- address already in use: another ui is running. Use `--port=8764` or stop the old one.
- new footage not in the grid: `scan` then `analyze`. The grid shows analyzed clips only.
- clip replaced on disk: the scan notices the changed size or mtime and re-queues it.
- a clip reads oversaturated in the audit: profile folders imply log. A non-log file in a log folder gets force-normalized. Move the file.
- a decode failure: lands as an error row in `status`, never a crash.
