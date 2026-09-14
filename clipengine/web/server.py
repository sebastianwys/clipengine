# server.py
# localhost-only web ui. stdlib http server: no frameworks, no external
# assets, nothing ever leaves 127.0.0.1. video previews stream straight
# from the footage tree with byte-range support so the browser can scrub.

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clipengine import catalog, config, matching, sequence

_STATIC = Path(__file__).parent / "static"
_CTYPES = {".mp4": "video/mp4", ".mov": "video/quicktime",
           ".mxf": "application/mxf"}
_CHUNK = 1024 * 1024
_THUMB_RE = re.compile(r"^/thumb/(\d+)/(start|mid|end)\.jpg$")
_MEDIA_RE = re.compile(r"^/media/(\d+)$")
_CLIP_RE = re.compile(r"^/api/clip/(\d+)$")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "ClipEngine/0.1"

    def log_message(self, fmt, *args):
        pass

    # -- plumbing ---------------------------------------------------------

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _bytes(self, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        route = url.path
        try:
            if route == "/":
                return self._bytes((_STATIC / "index.html").read_bytes(),
                                   "text/html; charset=utf-8")
            if route == "/api/overview":
                return self._overview()
            if route == "/api/clips":
                return self._clips(q)
            m = _CLIP_RE.match(route)
            if m:
                return self._clip(int(m.group(1)))
            if route == "/api/matches":
                return self._matches(q)
            if route == "/api/sequence":
                return self._sequence(q)
            m = _THUMB_RE.match(route)
            if m:
                return self._thumb(int(m.group(1)), m.group(2))
            m = _MEDIA_RE.match(route)
            if m:
                return self._media(int(m.group(1)))
            self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._error(500, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/export":
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                return self._export(payload)
            self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._error(500, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    # -- api ------------------------------------------------------------------

    def _overview(self):
        conn = catalog.connect()
        try:
            ov = catalog.overview(conn)
        finally:
            conn.close()
        ov["modes"] = list(config.SCORING_MODES)
        self._json(ov)

    def _clips(self, q):
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn, q.get("country") or None)
        finally:
            conn.close()
        klass = q.get("klass") or ""
        needle = (q.get("q") or "").lower()
        out = []
        for m in lib.meta:
            if klass and klass not in (m["start_class"], m["end_class"]):
                continue
            if needle and needle not in m["name"].lower():
                continue
            out.append({k: m[k] for k in
                        ("id", "name", "country", "profile", "duration_s",
                         "start_class", "end_class")})
        self._json({"clips": out})

    def _clip(self, clip_id: int):
        conn = catalog.connect()
        try:
            row = catalog.get_clip(conn, clip_id)
            if not row:
                return self._error(404, "unknown clip")
            feat = conn.execute(
                "SELECT summary, error FROM features WHERE clip_id=?",
                (clip_id,)).fetchone()
        finally:
            conn.close()
        detail = {"id": row["id"], "name": row["name"],
                  "country": row["country"], "profile": row["profile"],
                  "rel_path": row["rel_path"], "available": row["available"],
                  "size_bytes": row["size_bytes"],
                  "summary": json.loads(feat["summary"]) if feat and feat["summary"] else None,
                  "error": feat["error"] if feat else None}
        self._json(detail)

    def _matches(self, q):
        clip_id = int(q.get("clip", "0"))
        mode = q.get("mode", config.DEFAULT_MODE)
        n = int(q.get("n", "12"))
        country = q.get("country", "any")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        if clip_id not in lib.row_of:
            return self._error(400, "clip has no features yet")
        try:
            results = matching.rank(lib, clip_id, mode, n=n, country=country)
        except ValueError as exc:
            return self._error(400, str(exc))
        self._json({"clip": lib.meta[lib.row_of[clip_id]],
                    "mode": mode, "matches": results})

    def _sequence(self, q):
        seed = int(q.get("seed", "0"))
        mode = q.get("mode", config.DEFAULT_MODE)
        length = int(q.get("length", "8"))
        country_mode = q.get("country_mode", "any")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        if seed not in lib.row_of:
            return self._error(400, "seed clip has no features yet")
        matrix = matching.full_matrix(lib, mode)
        countries = [m["country"] for m in lib.meta]
        rows_idx, edges = sequence.build_chain(
            matrix, lib.row_of[seed], length,
            countries=countries, country_mode=country_mode)
        chain = [lib.meta[i] for i in rows_idx]
        self._json({"mode": mode, "total": round(sum(edges), 4),
                    "edges": [round(e, 4) for e in edges],
                    "clips": [{k: m[k] for k in
                               ("id", "name", "country", "duration_s",
                                "start_class", "end_class")}
                              for m in chain]})

    def _export(self, payload):
        ids = [int(i) for i in payload.get("ids", [])]
        mode = payload.get("mode", config.DEFAULT_MODE)
        if len(ids) < 2:
            return self._error(400, "need at least two clips to export")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        missing = [i for i in ids if i not in lib.row_of]
        if missing:
            return self._error(400, f"clips without features: {missing}")
        edges = []
        for a, b in zip(ids, ids[1:]):
            scores, _ = matching.score_against(lib, lib.row_of[a], mode)
            edges.append(float(scores[lib.row_of[b]]))
        meta = [lib.meta[lib.row_of[i]] for i in ids]
        json_path, m3u_path, xml_path = sequence.export_chain(
            meta, edges, mode)
        self._json({"json": str(json_path), "m3u8": str(m3u_path),
                    "fcpxml": str(xml_path),
                    "total": round(sum(edges), 4)})

    # -- files -------------------------------------------------------------------

    def _thumb(self, clip_id: int, pos: str):
        path = config.THUMB_DIR / f"{clip_id}_{pos}.jpg"
        if not path.exists():
            return self._error(404, "no thumbnail")
        self._bytes(path.read_bytes(), "image/jpeg", cache="max-age=86400")

    def _media(self, clip_id: int):
        """stream a clip with http range support. ids come from the
        catalog, so only cataloged footage paths are ever served, and an
        evicted file is refused rather than silently pulled from icloud."""
        conn = catalog.connect()
        try:
            row = catalog.get_clip(conn, clip_id)
        finally:
            conn.close()
        if not row:
            return self._error(404, "unknown clip")
        path = Path(row["path"])
        try:
            st = os.stat(path)
        except OSError:
            return self._error(404, "file missing on disk")
        if st.st_blocks == 0 and st.st_size > 0:
            return self._error(409, "clip is icloud-evicted;"
                               " download it in finder first")
        size = st.st_size
        ctype = _CTYPES.get(path.suffix.lower(), "application/octet-stream")
        start, end = 0, size - 1
        header = self.headers.get("Range")
        is_partial = False
        if header:
            m = _RANGE_RE.match(header.strip())
            if not m or (not m.group(1) and not m.group(2)):
                return self._error(416, "bad range")
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:
                # suffix form: last n bytes
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            is_partial = True
        self.send_response(206 if is_partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(port: int = config.SERVER_PORT) -> None:
    config.create_directories()
    httpd = ThreadingHTTPServer((config.SERVER_HOST, port), Handler)
    print(f"clipengine ui: http://{config.SERVER_HOST}:{port} (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
