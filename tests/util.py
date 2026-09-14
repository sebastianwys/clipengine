# util.py
# shared test scaffolding. TempDirsMixin repoints every config path at a
# throwaway directory so tests can never touch the real catalog, real
# thumbnails, or real exports.

import tempfile
from pathlib import Path

from clipengine import config


class TempDirsMixin:
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="clipengine-test-")
        self.tmp = Path(self._tmpdir.name)
        self._saved = (config.DATA_DIR, config.DB_PATH,
                       config.THUMB_DIR, config.EXPORT_DIR)
        config.DATA_DIR = self.tmp / "data"
        config.DB_PATH = config.DATA_DIR / "catalog.db"
        config.THUMB_DIR = config.DATA_DIR / "thumbs"
        config.EXPORT_DIR = self.tmp / "exports"
        config.create_directories()

    def tearDown(self):
        (config.DATA_DIR, config.DB_PATH,
         config.THUMB_DIR, config.EXPORT_DIR) = self._saved
        self._tmpdir.cleanup()
        super().tearDown()
