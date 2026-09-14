# catalog.py
# sqlite catalog of every clip under the footage tree. the scan is
# stat-only: it never opens a video file, so icloud-evicted stubs are
# cataloged without triggering a single byte of download.

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clipengine import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    profile TEXT NOT NULL,
    is_log INTEGER NOT NULL,
    country TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_key TEXT NOT NULL,
    available INTEGER NOT NULL,
    oversize INTEGER NOT NULL DEFAULT 0,
    missing INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_country ON clips(country);
CREATE INDEX IF NOT EXISTS idx_clips_available ON clips(available);

CREATE TABLE IF NOT EXISTS features (
    clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content_key TEXT NOT NULL,
    vector BLOB,
    summary TEXT,
    error TEXT,
    analyzed_at TEXT NOT NULL
);
"""

# join condition for a feature row that is still valid for its clip:
# same algorithm version and the file has not changed since analysis
_FRESH = ("f.clip_id = c.id AND f.version = ? AND f.content_key = c.content_key"
          " AND f.error IS NULL AND f.vector IS NOT NULL")


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """open the catalog, creating directories and schema on first use.
    the path resolves at call time, not import time, so tests can
    repoint config at a temp directory."""
    if db_path is None:
        db_path = config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    # microsecond precision: scan runs compare these strings to find rows
    # the latest walk did not touch, so two scans in the same second must
    # still produce distinct stamps
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def is_materialized(st: os.stat_result) -> bool:
    """true when the file has local data. icloud-evicted stubs report full
    size but occupy zero blocks; reading one forces a network download."""
    return st.st_blocks > 0 and st.st_size > 0


def classify_path(rel: Path) -> tuple[str, int, str]:
    """map a media-root-relative path onto (profile, is_log, country).
    layout is <camera profile>/<country>/file, so country is the vibe tag."""
    parts = rel.parts
    profile_key, is_log = "unknown", 0
    country = "Unsorted"
    if len(parts) >= 1:
        cam = config.CAMERA_PROFILES.get(parts[0])
        if cam:
            profile_key = cam["key"]
            is_log = 1 if cam["log"] else 0
        else:
            profile_key = parts[0]
    if len(parts) >= 3:
        country = parts[1]
    return profile_key, is_log, country


def scan(conn: sqlite3.Connection, media_root: Optional[Path] = None) -> dict:
    """walk the footage tree with stat calls only and upsert the catalog."""
    if media_root is None:
        media_root = config.MEDIA_ROOT
    if not media_root.exists():
        raise FileNotFoundError(f"media root not found: {media_root}")
    stats = {"seen": 0, "new": 0, "changed": 0, "available": 0,
             "evicted": 0, "oversize": 0, "missing": 0}
    now = _now()
    existing = {row["path"]: row for row in
                conn.execute("SELECT id, path, content_key FROM clips")}

    for dirpath, dirnames, filenames in os.walk(media_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in config.VIDEO_EXTENSIONS:
                continue
            full = Path(dirpath) / fname
            try:
                st = full.stat()
            except OSError as exc:
                logger.warning("stat failed for %s: %s", full, exc)
                continue
            rel = full.relative_to(media_root)
            profile, is_log, country = classify_path(rel)
            available = 1 if is_materialized(st) else 0
            oversize = 1 if st.st_size > config.MAX_ANALYZE_BYTES else 0
            key = f"{st.st_size}-{st.st_mtime_ns}"
            stats["seen"] += 1
            stats["available" if available else "evicted"] += 1
            stats["oversize"] += oversize
            row = existing.get(str(full))
            if row is None:
                conn.execute(
                    "INSERT INTO clips (path, rel_path, name, profile, is_log,"
                    " country, ext, size_bytes, mtime_ns, content_key, available,"
                    " oversize, missing, first_seen, last_seen)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
                    (str(full), str(rel), fname, profile, is_log, country, ext,
                     st.st_size, st.st_mtime_ns, key, available, oversize, now, now))
                stats["new"] += 1
            else:
                if row["content_key"] != key:
                    stats["changed"] += 1
                conn.execute(
                    "UPDATE clips SET size_bytes=?, mtime_ns=?, content_key=?,"
                    " available=?, oversize=?, missing=0, last_seen=? WHERE id=?",
                    (st.st_size, st.st_mtime_ns, key, available, oversize,
                     now, row["id"]))

    # anything not touched this pass is gone from disk
    cur = conn.execute(
        "UPDATE clips SET missing=1 WHERE missing=0 AND last_seen != ?", (now,))
    stats["missing"] = cur.rowcount
    conn.commit()
    return stats


# -- queries -----------------------------------------------------------

def overview(conn: sqlite3.Connection) -> dict:
    """headline counts plus a per-country rollup for status displays."""
    v = config.FEATURE_VERSION
    totals = conn.execute(
        f"""SELECT COUNT(*) AS total,
                   SUM(c.available) AS available,
                   SUM(CASE WHEN c.available=0 THEN 1 ELSE 0 END) AS evicted,
                   SUM(c.oversize) AS oversize,
                   SUM(CASE WHEN f.clip_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed
            FROM clips c LEFT JOIN features f ON {_FRESH}
            WHERE c.missing=0""", (v,)).fetchone()
    errors = conn.execute(
        "SELECT COUNT(*) FROM features WHERE error IS NOT NULL").fetchone()[0]
    countries = [dict(r) for r in conn.execute(
        f"""SELECT c.country,
                   COUNT(*) AS total,
                   SUM(c.available) AS available,
                   SUM(CASE WHEN f.clip_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed
            FROM clips c LEFT JOIN features f ON {_FRESH}
            WHERE c.missing=0
            GROUP BY c.country ORDER BY c.country""", (v,))]
    profiles = [dict(r) for r in conn.execute(
        """SELECT profile, COUNT(*) AS total, SUM(available) AS available
           FROM clips WHERE missing=0 GROUP BY profile ORDER BY profile""")]
    return {"totals": dict(totals), "errors": errors,
            "countries": countries, "profiles": profiles}


def pending(conn: sqlite3.Connection, country: Optional[str] = None,
            limit: Optional[int] = None, force: bool = False) -> list[sqlite3.Row]:
    """clips that can and should be analyzed: locally materialized, not
    oversize, and lacking a fresh feature row (unless force)."""
    v = config.FEATURE_VERSION
    sql = (f"SELECT c.* FROM clips c LEFT JOIN features f ON {_FRESH}"
           " WHERE c.available=1 AND c.missing=0 AND c.oversize=0")
    params: list = [v]
    if not force:
        sql += " AND f.clip_id IS NULL"
    if country:
        sql += " AND c.country = ?"
        params.append(country)
    sql += " ORDER BY c.country, c.name"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def analyzed(conn: sqlite3.Connection,
             country: Optional[str] = None) -> list[sqlite3.Row]:
    """clips with a fresh feature vector, joined with the vector itself."""
    v = config.FEATURE_VERSION
    sql = (f"SELECT c.*, f.vector AS vector, f.summary AS summary"
           f" FROM clips c JOIN features f ON {_FRESH}"
           " WHERE c.missing=0")
    params: list = [v]
    if country:
        sql += " AND c.country = ?"
        params.append(country)
    sql += " ORDER BY c.id"
    return conn.execute(sql, params).fetchall()


def save_features(conn: sqlite3.Connection, clip_id: int, content_key: str,
                  vector: Optional[bytes], summary: Optional[str],
                  error: Optional[str] = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO features"
        " (clip_id, version, content_key, vector, summary, error, analyzed_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (clip_id, config.FEATURE_VERSION, content_key, vector, summary,
         error, _now()))
    conn.commit()


def get_clip(conn: sqlite3.Connection, clip_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()


def find_clip(conn: sqlite3.Connection, token: str) -> Optional[sqlite3.Row]:
    """resolve a cli argument: numeric id first, then name substring."""
    if token.isdigit():
        row = get_clip(conn, int(token))
        if row:
            return row
    return conn.execute(
        "SELECT * FROM clips WHERE name LIKE ? AND missing=0 ORDER BY id LIMIT 1",
        (f"%{token}%",)).fetchone()
