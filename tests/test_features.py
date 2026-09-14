# test_features.py
# the vector layout is the contract between analyzer and matcher, so it
# gets its own tests: unique names, exact round trips, loud failures.

import unittest

import numpy as np

from clipengine import features


def full_scalars(value: float = 1.0) -> dict:
    return {name: value for name in features.FIELDS
            if not name.startswith(("start_hue_", "end_hue_"))}


class TestLayout(unittest.TestCase):
    def test_no_duplicate_fields(self):
        self.assertEqual(len(set(features.FIELDS)), features.VECTOR_LEN)

    def test_index_matches_fields(self):
        for i, name in enumerate(features.FIELDS):
            self.assertEqual(features.INDEX[name], i)

    def test_hue_slices_cover_bins(self):
        self.assertEqual(features.START_HUE.stop - features.START_HUE.start,
                         features.HUE_BINS)
        self.assertEqual(features.END_HUE.stop - features.END_HUE.start,
                         features.HUE_BINS)


class TestPackUnpack(unittest.TestCase):
    def test_roundtrip(self):
        scalars = {name: float(i) for i, name in enumerate(full_scalars())}
        start_hue = np.linspace(0, 1, features.HUE_BINS)
        end_hue = np.linspace(1, 0, features.HUE_BINS)
        vec = features.pack(scalars, start_hue, end_hue)
        blob = features.to_bytes(vec)
        back = features.from_bytes(blob)
        np.testing.assert_allclose(back, vec)
        for name, value in scalars.items():
            self.assertAlmostEqual(features.get(back, name), value, places=4)
        np.testing.assert_allclose(back[features.START_HUE], start_hue,
                                   atol=1e-6)
        np.testing.assert_allclose(back[features.END_HUE], end_hue, atol=1e-6)

    def test_missing_scalar_raises(self):
        scalars = full_scalars()
        scalars.pop("end_energy")
        hue = np.zeros(features.HUE_BINS)
        with self.assertRaises(KeyError):
            features.pack(scalars, hue, hue)

    def test_wrong_blob_length_raises(self):
        with self.assertRaises(ValueError):
            features.from_bytes(b"\x00" * 12)


if __name__ == "__main__":
    unittest.main()
