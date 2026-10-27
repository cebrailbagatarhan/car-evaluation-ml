"""Offline tests for the Car Evaluation dataset explorer."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from explore_dataset import main, summarize


class ExploreDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cars.csv"
        self.path.write_text(
            "low,low,4,more,big,high,vgood\n"
            "high,low,2,2,small,low,unacc\n"
            "low,med,4,more,med,high,good\n", encoding="utf-8"
        )

    def test_filters_and_class_distribution(self):
        self.assertEqual(
            summarize(self.path, {"safety": "high", "buying": "low"}),
            {"matched": 2, "filters": {"safety": "high", "buying": "low"}, "classes": {"good": 1, "vgood": 1}},
        )

    def test_no_matches(self):
        self.assertEqual(summarize(self.path, {"persons": "4"})["matched"], 0)

    def test_json_cli_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--data", str(self.path), "--safety", "low", "--format", "json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["classes"], {"unacc": 1})

    def test_bad_row_rejected(self):
        self.path.write_text("low,low\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 1"):
            summarize(self.path, {})

    def test_missing_file_exits_nonzero(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--data", str(self.path) + ".missing"])
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
