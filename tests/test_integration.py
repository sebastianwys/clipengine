# test_integration.py
# the whole pipeline on a synthetic library: scan -> analyze -> match ->
# chain -> export, exactly as the cli drives it. six clips across three
# camera profiles and four countries, each with a known motion signature.

import json
import unittest

from clipengine import catalog, cli, config, matching, sequence
from tests import synth
from tests.util import TempDirsMixin


class TestEndToEnd(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        spec = {
            ("Sony SLOG-3", "Japan", "pan_a.mp4"): ["pan_right", "pan_right"],
            ("Sony SLOG-3", "Japan", "pan_b.mp4"): ["pan_right", "pan_right"],
            ("Sony SLOG-3", "Iceland", "still.mp4"): ["static", "static"],
            ("DJI DLOG-M", "Italy", "whip.mp4"): ["whip_right", "whip_right"],
            ("DJI DLOG-M", "Italy", "zoom.mp4"): ["push_in", "push_in"],
            ("Apple ProRes-Log", "Australia", "calm.mp4"): ["static", "static"],
        }
        for (profile, country, name), kinds in spec.items():
            path = self.root / profile / country / name
            path.parent.mkdir(parents=True, exist_ok=True)
            px = 14 if kinds[0].startswith("whip") else 3
            synth.make_clip(path, kinds, n_each=36, px=px)

    def test_pipeline(self):
        conn = catalog.connect()
        self.addCleanup(conn.close)
        stats = catalog.scan(conn, self.root)
        self.assertEqual(stats["seen"], 6)
        self.assertEqual(stats["available"], 6)
        self.assertEqual(len(catalog.pending(conn)), 6)

        cli.cmd_analyze([])  # drives the real analyzer over the catalog

        self.assertEqual(len(catalog.pending(conn)), 0)
        analyzed = catalog.analyzed(conn)
        self.assertEqual(len(analyzed), 6)
        self.assertEqual(len(list(config.THUMB_DIR.glob("*.jpg"))), 18)

        summaries = {r["name"]: json.loads(r["summary"]) for r in analyzed}
        self.assertEqual(summaries["pan_a.mp4"]["end_class"], "pan_right")
        self.assertEqual(summaries["pan_b.mp4"]["start_class"], "pan_right")
        self.assertTrue(
            summaries["whip.mp4"]["start_class"].startswith("whip"))
        self.assertEqual(summaries["zoom.mp4"]["start_class"], "push_in")
        self.assertEqual(summaries["still.mp4"]["start_class"], "static")

        lib = matching.load_library(conn)
        self.assertEqual(len(lib), 6)
        pan_a = catalog.find_clip(conn, "pan_a")
        pan_b = catalog.find_clip(conn, "pan_b")
        results = matching.rank(lib, pan_a["id"], "momentum", n=5)
        self.assertIn(pan_b["id"], [r["id"] for r in results[:2]])
        by_id = {r["id"]: r for r in results}
        self.assertGreater(by_id[pan_b["id"]]["breakdown"]["motion"], 0.85)

        matrix = matching.full_matrix(lib, "momentum")
        rows, edges = sequence.build_chain(
            matrix, lib.row_of[pan_a["id"]], length=4)
        self.assertEqual(len(rows), len(set(rows)))
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(e > 0 for e in edges))

        meta = [lib.meta[i] for i in rows]
        json_path, m3u_path, xml_path = sequence.export_chain(
            meta, edges, "momentum")
        plan = json.loads(json_path.read_text())
        self.assertEqual(len(plan["clips"]), len(rows))
        self.assertIn(meta[0]["path"], m3u_path.read_text())
        self.assertIn("asset-clip", xml_path.read_text())


if __name__ == "__main__":
    unittest.main()
