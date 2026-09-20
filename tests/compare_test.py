import unittest
from compare_evqa import compare_rows


class ComparisonTest(unittest.TestCase):
    def test_equal(self):
        rows = [["0", "0", "1", "0.25"]]
        self.assertEqual(compare_rows(rows, rows)["max_score_error"], 0)

    def test_invalid(self):
        good = [["0", "0", "1", "0.25"]]
        for score in ("nan", "inf", "-inf", "0.5"):
            bad = [["0", "0", "1", score]]
            with self.assertRaises(RuntimeError):
                compare_rows(bad, good)
            with self.assertRaises(RuntimeError):
                compare_rows(good, bad)
        for rows in ([], [["0", "0", "1"]], good + good):
            with self.assertRaises(RuntimeError):
                compare_rows(rows, good)
        with self.assertRaises(RuntimeError):
            compare_rows(good + good, good + good)


if __name__ == "__main__":
    unittest.main()
