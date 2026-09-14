# test_relabel.py
# relabel recomputes motion labels from stored vectors without touching
# any video file. that is the mechanism that makes threshold tuning
# cheap: change config, run relabel, done. the audit dashboard shares
# the fixture since both read the same stored vectors.

import contextlib
import io
import json
import unittest

from clipengine import catalog, cli, features
from tests.test_matching import make_vec
from tests.util import TempDirsMixin


class TestRelabel(TempDirsMixin, unittest.TestCase):
    def _insert_clip(self, conn, name: str) -> int:
        cur = conn.execute(
            "INSERT INTO clips (path, rel_path, name, profile, is_log,"
            " country, ext, size_bytes, mtime_ns, content_key, available,"
            " oversize, missing, first_seen, last_seen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            (f"/x/{name}", name, name, "t", 0, "X", ".mp4",
             1, 1, "1-1", 1, 0, "t", "t"))
        return cur.lastrowid

    def test_stale_labels_get_recomputed(self):
        conn = catalog.connect()
        clip_id = self._insert_clip(conn, "a.mp4")
        vec = make_vec(end_flow_x=-0.5, end_energy=0.5, end_jitter=0.05)
        stale = {"start_class": "pan_left", "end_class": "static",
                 "start": {"class": "pan_left"}, "end": {"class": "static"},
                 "duration_s": 5.0}
        catalog.save_features(conn, clip_id, "1-1",
                              features.to_bytes(vec), json.dumps(stale))
        conn.close()

        cli.cmd_relabel([])

        conn = catalog.connect()
        summary = json.loads(conn.execute(
            "SELECT summary FROM features WHERE clip_id=?",
            (clip_id,)).fetchone()["summary"])
        conn.close()
        self.assertEqual(summary["start_class"], "static")
        self.assertEqual(summary["end_class"], "pan_right")
        self.assertEqual(summary["start"]["class"], "static")
        self.assertEqual(summary["end"]["class"], "pan_right")
        self.assertEqual(summary["duration_s"], 5.0)  # untouched fields survive

    def test_relabel_is_idempotent(self):
        conn = catalog.connect()
        clip_id = self._insert_clip(conn, "b.mp4")
        vec = make_vec()
        catalog.save_features(
            conn, clip_id, "1-1", features.to_bytes(vec),
            json.dumps({"start_class": "static", "end_class": "static"}))
        conn.close()
        cli.cmd_relabel([])
        cli.cmd_relabel([])
        conn = catalog.connect()
        summary = json.loads(conn.execute(
            "SELECT summary FROM features WHERE clip_id=?",
            (clip_id,)).fetchone()["summary"])
        conn.close()
        self.assertEqual(summary["start_class"], "static")


class TestAudit(TempDirsMixin, unittest.TestCase):
    def test_audit_reports_all_sections(self):
        conn = catalog.connect()
        for name, vec in (
                ("a.mp4", make_vec(end_flow_x=-0.5, end_energy=0.5)),
                ("b.mp4", make_vec(start_flow_x=0.3, start_energy=0.3,
                                   start_jitter=0.8))):
            cur = conn.execute(
                "INSERT INTO clips (path, rel_path, name, profile, is_log,"
                " country, ext, size_bytes, mtime_ns, content_key,"
                " available, oversize, missing, first_seen, last_seen)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
                (f"/x/{name}", name, name, "t", 1, "Japan", ".mp4",
                 1, 1, "1-1", 1, 0, "t", "t"))
            catalog.save_features(
                conn, cur.lastrowid, "1-1", features.to_bytes(vec),
                json.dumps({"start_class": "static", "end_class": "static",
                            "flat": True}))
        conn.close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.cmd_audit([])
        text = out.getvalue()
        self.assertIn("audit over 2 analyzed clips", text)
        self.assertIn("class distribution", text)
        self.assertIn("window energy", text)
        self.assertIn("steadiness", text)
        self.assertIn("flat/log detected: 2/2", text)
        self.assertIn("Japan", text)
        self.assertIn("no analysis errors", text)

    def test_audit_empty_catalog(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.cmd_audit([])
        self.assertIn("nothing analyzed yet", out.getvalue())


if __name__ == "__main__":
    unittest.main()
