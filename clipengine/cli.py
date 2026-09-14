# cli.py
# entry point. hand-rolled sys.argv dispatch, no argparse.
# heavy imports (cv2, numpy paths) stay inside the commands that need
# them so `status` and `help` start instantly.

import json
import sys
import time
from pathlib import Path

from clipengine import catalog, config

HELP = """clipengine: local footage transition-matching engine

usage: python -m clipengine <command>

commands:
  scan [--root=PATH]          stat-only walk of the footage tree into the
                              catalog. never opens files, never downloads.
  status                      catalog counts by country and profile
  analyze [--limit=N] [--country=X] [--force]
                              extract features from materialized clips
  relabel                     recompute motion labels from stored vectors
                              after threshold changes; no video decoding
  audit                       tuning dashboard: class distribution, energy
                              percentiles vs thresholds, per-country vibe
  match <id|name> [--mode=M] [--n=10] [--country=any|same|different]
                              rank transition candidates out of one clip
  sequence --seed=<id|name> [--length=8] [--mode=M]
           [--country-mode=any|same|travel] [--no-export]
                              build a best chain and export json + m3u8
  ui [--port=8763]            launch the local web ui (127.0.0.1 only)

modes: momentum (carry motion through the cut), whip (blur to blur),
calm (still to still, color led), contrast (deliberate vibe flip)
"""


def _opt(args: list[str], name: str, default=None):
    """--name=value style option."""
    prefix = f"--{name}="
    for a in args:
        if a.startswith(prefix):
            return a[len(prefix):]
    return default


def _flag(args: list[str], name: str) -> bool:
    return f"--{name}" in args


def _print_countries(conn) -> None:
    ov = catalog.overview(conn)
    print(f"\n  {'country':<24} {'clips':>6} {'local':>6} {'analyzed':>9}")
    for c in ov["countries"]:
        print(f"  {c['country']:<24} {c['total']:>6}"
              f" {c['available'] or 0:>6} {c['analyzed'] or 0:>9}")


def cmd_scan(args: list[str]) -> None:
    conn = catalog.connect()
    root = _opt(args, "root")
    media_root = Path(root) if root else config.MEDIA_ROOT
    stats = catalog.scan(conn, media_root)
    print(f"scan of {media_root}")
    print(f"  seen {stats['seen']} videos | new {stats['new']}"
          f" | changed {stats['changed']} | gone missing {stats['missing']}")
    print(f"  materialized {stats['available']}"
          f" | icloud-evicted {stats['evicted']}"
          f" | oversize-skipped {stats['oversize']}")
    _print_countries(conn)
    conn.close()


def cmd_status(args: list[str]) -> None:
    conn = catalog.connect()
    ov = catalog.overview(conn)
    t = ov["totals"]
    print(f"catalog: {t['total'] or 0} clips | {t['available'] or 0} local"
          f" | {t['evicted'] or 0} evicted | {t['analyzed'] or 0} analyzed"
          f" | {ov['errors']} errors")
    for p in ov["profiles"]:
        print(f"  {p['profile']:<20} {p['total']:>5} clips"
              f" ({p['available'] or 0} local)")
    _print_countries(conn)
    conn.close()


def cmd_analyze(args: list[str]) -> None:
    from clipengine import analysis, features
    conn = catalog.connect()
    config.create_directories()
    limit = _opt(args, "limit")
    rows = catalog.pending(conn,
                           country=_opt(args, "country"),
                           limit=int(limit) if limit else None,
                           force=_flag(args, "force"))
    if not rows:
        print("nothing to analyze: every available clip has fresh features")
        return
    print(f"analyzing {len(rows)} clip(s)")
    done = failed = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        label = f"[{i}/{len(rows)}] {row['country']}/{row['name']}"
        t1 = time.time()
        try:
            result = analysis.analyze_clip(row["path"], bool(row["is_log"]))
            for pos, blob in result.thumbs.items():
                (config.THUMB_DIR / f"{row['id']}_{pos}.jpg").write_bytes(blob)
            catalog.save_features(conn, row["id"], row["content_key"],
                                  features.to_bytes(result.vector),
                                  json.dumps(result.summary))
            s = result.summary
            print(f"{label}: {s['start_class']} -> {s['end_class']}"
                  f" | {s['duration_s']}s | {time.time() - t1:.1f}s")
            done += 1
        except Exception as exc:
            catalog.save_features(conn, row["id"], row["content_key"],
                                  None, None, error=str(exc))
            print(f"{label}: FAILED ({exc})")
            failed += 1
    print(f"done: {done} analyzed, {failed} failed,"
          f" {time.time() - t0:.0f}s total")
    conn.close()


def cmd_relabel(args: list[str]) -> None:
    from clipengine import analysis, features
    conn = catalog.connect()
    rows = conn.execute("SELECT clip_id, vector, summary FROM features"
                        " WHERE vector IS NOT NULL").fetchall()
    changed = 0
    for row in rows:
        try:
            vec = features.from_bytes(row["vector"])
        except ValueError:
            continue
        summary = json.loads(row["summary"] or "{}")
        start = analysis.motion_from_vector(vec, "start")
        end = analysis.motion_from_vector(vec, "end")
        if (summary.get("start_class") == start.label
                and summary.get("end_class") == end.label):
            continue
        summary["start_class"] = start.label
        summary["end_class"] = end.label
        if isinstance(summary.get("start"), dict):
            summary["start"]["class"] = start.label
        if isinstance(summary.get("end"), dict):
            summary["end"]["class"] = end.label
        conn.execute("UPDATE features SET summary=? WHERE clip_id=?",
                     (json.dumps(summary), row["clip_id"]))
        changed += 1
    conn.commit()
    print(f"relabeled {changed} of {len(rows)} analyzed clips")
    conn.close()


def cmd_audit(args: list[str]) -> None:
    """distribution report over analyzed clips. the tuning dashboard:
    run after big analyze passes or threshold changes and check that the
    thresholds carve the real distribution at sensible points."""
    import json as _json
    from collections import Counter

    import numpy as np

    from clipengine import features, matching
    conn = catalog.connect()
    lib = matching.load_library(conn)
    if not len(lib):
        print("nothing analyzed yet")
        conn.close()
        return
    F, idx, n = lib.F, features.INDEX, len(lib)
    print(f"audit over {n} analyzed clips")

    classes = Counter()
    for m in lib.meta:
        classes[m["start_class"]] += 1
        classes[m["end_class"]] += 1
    print("\nclass distribution (start + end windows):")
    for name, cnt in classes.most_common():
        bar = "#" * max(1, round(40 * cnt / (2 * n)))
        print(f"  {str(name):<14} {cnt:>4}  {bar}")

    energy = np.concatenate([F[:, idx["start_energy"]],
                             F[:, idx["end_energy"]]])
    pct = np.percentile(energy, [10, 25, 50, 75, 90, 99])
    print("\nwindow energy, widths/sec (p10 p25 p50 p75 p90 p99):")
    print("  " + "  ".join(f"{v:.3f}" for v in pct))
    print(f"  thresholds: static < {config.STATIC_MAX}"
          f" | pan >= {config.PAN_MIN} | whip >= {config.WHIP_MIN}")

    tx = np.concatenate([F[:, idx["start_flow_x"]], F[:, idx["end_flow_x"]]])
    ty = np.concatenate([F[:, idx["start_flow_y"]], F[:, idx["end_flow_y"]]])
    jit = np.concatenate([F[:, idx["start_jitter"]], F[:, idx["end_jitter"]]])
    t = np.hypot(tx, ty)
    eligible = t >= config.PAN_MIN
    ratio = t[eligible] / (jit[eligible] + 1e-6)
    steady = ratio >= config.STEADY_RATIO
    near = ((ratio >= config.STEADY_RATIO * 0.75)
            & (ratio <= config.STEADY_RATIO * 1.25))
    print(f"\nsteadiness (windows with pan-level translation):"
          f" {int(eligible.sum())} eligible,"
          f" {int(steady.sum())} steady, {int((~steady).sum())} handheld,"
          f" {int(near.sum())} near the boundary")

    flat = 0
    for row in catalog.analyzed(conn):
        summary = _json.loads(row["summary"] or "{}")
        flat += 1 if summary.get("flat") else 0
    print(f"\nflat/log detected: {flat}/{n} clips"
          " (expect nearly all: every profile shoots log)")

    print("\nper-country vibe (mean luma / sat / warmth, count):")
    by_country: dict = {}
    for i, m in enumerate(lib.meta):
        by_country.setdefault(m["country"], []).append(i)
    for country in sorted(by_country):
        rows = by_country[country]
        luma = float(F[rows, idx["global_luma"]].mean())
        sat = float(F[rows, idx["global_sat"]].mean())
        warm = float(F[rows, idx["global_warmth"]].mean())
        print(f"  {country:<22} luma {luma:5.1f}  sat {sat:5.1f}"
              f"  warmth {warm:+5.1f}  ({len(rows)})")

    errors = conn.execute(
        "SELECT c.name, f.error FROM features f JOIN clips c"
        " ON c.id = f.clip_id WHERE f.error IS NOT NULL").fetchall()
    if errors:
        print(f"\n{len(errors)} analysis errors:")
        for e in errors[:10]:
            print(f"  {e['name']}: {e['error']}")
    else:
        print("\nno analysis errors")
    conn.close()


def cmd_match(args: list[str]) -> None:
    from clipengine import matching
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print("usage: match <clip-id-or-name> [--mode=] [--n=] [--country=]")
        return
    conn = catalog.connect()
    row = catalog.find_clip(conn, positional[0])
    if not row:
        print(f"no clip matches '{positional[0]}'")
        return
    mode = _opt(args, "mode", config.DEFAULT_MODE)
    lib = matching.load_library(conn)
    if row["id"] not in lib.row_of:
        print(f"clip #{row['id']} ({row['name']}) has no features yet;"
              " run analyze first")
        return
    a = lib.meta[lib.row_of[row["id"]]]
    print(f"cuts out of #{row['id']} {row['country']}/{row['name']}"
          f" (end: {a['end_class']}) mode={mode}")
    results = matching.rank(lib, row["id"], mode,
                            n=int(_opt(args, "n", "10")),
                            country=_opt(args, "country", "any"))
    if not results:
        print("no positive-score candidates under this mode's gate")
        return
    for r_i, r in enumerate(results, 1):
        b = r["breakdown"]
        print(f"{r_i:>2}. {r['score']:.3f}  #{r['id']:<4}"
              f" {r['country']}/{r['name']}  starts:{r['start_class']}"
              f"  (motion {b['motion']}, energy {b['energy']},"
              f" color {b['color']}, gate {b['gate']})")
    conn.close()


def cmd_sequence(args: list[str]) -> None:
    from clipengine import matching, sequence
    seed_token = _opt(args, "seed")
    if not seed_token:
        print("usage: sequence --seed=<id-or-name> [--length=] [--mode=]"
              " [--country-mode=] [--no-export]")
        return
    conn = catalog.connect()
    row = catalog.find_clip(conn, seed_token)
    if not row:
        print(f"no clip matches '{seed_token}'")
        return
    lib = matching.load_library(conn)
    if row["id"] not in lib.row_of:
        print(f"seed #{row['id']} has no features yet; run analyze first")
        return
    mode = _opt(args, "mode", config.DEFAULT_MODE)
    matrix = matching.full_matrix(lib, mode)
    countries = [m["country"] for m in lib.meta]
    rows_idx, edges = sequence.build_chain(
        matrix, lib.row_of[row["id"]],
        length=int(_opt(args, "length", "8")),
        countries=countries,
        country_mode=_opt(args, "country-mode", "any"))
    meta = [lib.meta[i] for i in rows_idx]
    print(f"sequence ({mode}), total score {sum(edges):.3f}")
    for i, m in enumerate(meta):
        cut = "  seed" if i == 0 else f"{edges[i - 1]:.3f}"
        print(f" {i + 1:>2}. [{cut:>6}] #{m['id']:<4} {m['country']}/{m['name']}"
              f" ({m['start_class']} -> {m['end_class']})")
    if not _flag(args, "no-export"):
        json_path, m3u_path, xml_path = sequence.export_chain(
            meta, edges, mode)
        print(f"exported: {json_path}")
        print(f"          {m3u_path}")
        print(f"          {xml_path}  (import into fcp or resolve)")
    conn.close()


def cmd_ui(args: list[str]) -> None:
    from clipengine.web import server
    server.serve(int(_opt(args, "port", str(config.SERVER_PORT))))


def main(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("help", "-h", "--help"):
        print(HELP)
        return
    commands = {"scan": cmd_scan, "status": cmd_status,
                "analyze": cmd_analyze, "relabel": cmd_relabel,
                "audit": cmd_audit, "match": cmd_match,
                "sequence": cmd_sequence, "ui": cmd_ui}
    fn = commands.get(args[0])
    if fn is None:
        print(f"unknown command: {args[0]}")
        print(HELP)
        sys.exit(1)
    fn(args[1:])
