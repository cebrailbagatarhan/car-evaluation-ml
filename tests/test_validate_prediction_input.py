"""Regression tests for the dependency-free batch input validator."""

import json
import tempfile
import unittest
from pathlib import Path

from validate_prediction_input import main, validate_file


VALID = {
    "buying": "high",
    "maint": "med",
    "doors": "4",
    "persons": "4",
    "lug_boot": "big",
    "safety": "high",
}


class BatchInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, contents):
        path = self.root / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_valid_json(self):
        path = self.write("input.json", json.dumps([VALID, VALID]))
        self.assertEqual(validate_file(path)["valid_rows"], 2)
        self.assertEqual(main([str(path), "--format", "json"]), 0)

    def test_valid_csv(self):
        path = self.write(
            "input.csv",
            "buying,maint,doors,persons,lug_boot,safety\n"
            "high,med,4,4,big,high\n",
        )
        report = validate_file(path)
        self.assertEqual((report["rows"], report["valid_rows"], report["errors"]), (1, 1, []))

    def test_invalid_value(self):
        path = self.write("input.json", json.dumps([{**VALID, "safety": "unsafe"}]))
        report = validate_file(path)
        self.assertEqual(report["valid_rows"], 0)
        self.assertIn("invalid safety", report["errors"][0])

    def test_missing_and_extra_features(self):
        path = self.write("input.json", json.dumps([{"buying": "high", "other": "x"}]))
        report = validate_file(path)
        self.assertTrue(any("missing columns" in error for error in report["errors"]))
        self.assertTrue(any("unexpected columns" in error for error in report["errors"]))

    def test_reject_non_object_json_items(self):
        path = self.write("input.json", json.dumps([VALID, 123]))
        report = validate_file(path)
        self.assertEqual((report["rows"], report["valid_rows"]), (2, 1))
        self.assertIn("expected an object", report["errors"][0])

    def test_reject_duplicate_csv_headers(self):
        path = self.write("input.csv", "buying,buying,doors,persons,lug_boot,safety\n")
        self.assertIn("duplicate", validate_file(path)["errors"][0])

    def test_reject_extra_csv_fields(self):
        path = self.write("input.csv", "buying,maint,doors,persons,lug_boot,safety\n"
                                     "high,med,4,4,big,high,unexpected\n")
        self.assertIn("too many fields", validate_file(path)["errors"][0])

    def test_reject_empty_json_array(self):
        path = self.write("input.json", "[]")
        self.assertIn("no prediction rows", validate_file(path)["errors"][0])

    def test_reject_invalid_json_and_missing_file(self):
        path = self.write("input.json", "{broken")
        self.assertIn("Cannot read", validate_file(path)["errors"][0])
        self.assertIn("Cannot read", validate_file(self.root / "missing.csv")["errors"][0])

    def test_reject_unsupported_extension(self):
        path = self.write("input.txt", "test")
        self.assertIn("Unsupported", validate_file(path)["errors"][0])


if __name__ == "__main__":
    unittest.main()

    def test_jsonl_accepts_one_record_per_line_and_blank_lines(self):
        path = self.write("input.jsonl", "\n".join([
            json.dumps(VALID), "", "  ", json.dumps(VALID), "",
        ]))
        report = validate_file(path)
        self.assertEqual((report["rows"], report["valid_rows"], report["errors"]), (2, 2, []))
        self.assertEqual(main([str(path), "--format", "json"]), 0)

    def test_ndjson_reports_bad_lines_and_continues(self):
        path = self.write("input.ndjson", "\n".join([
            json.dumps(VALID), "{bad", json.dumps({**VALID, "safety": "unsafe"}),
            json.dumps(VALID),
        ]))
        report = validate_file(path)
        self.assertEqual((report["rows"], report["valid_rows"]), (4, 2))
        self.assertEqual(len(report["errors"]), 2)
        self.assertIn("JSONL line 2: invalid JSON", report["errors"][0])
        self.assertIn("JSONL line 3: invalid safety", report["errors"][1])
        self.assertEqual(main([str(path)]), 1)

    def test_jsonl_rejects_non_object_items(self):
        path = self.write("input.jsonl", json.dumps(123) + "\n")
        report = validate_file(path)
        self.assertEqual((report["rows"], report["valid_rows"]), (1, 0))
        self.assertIn("expected an object", report["errors"][0])

    def test_jsonl_with_only_blank_lines_is_empty(self):
        path = self.write("input.jsonl", "\n   \n")
        report = validate_file(path)
        self.assertEqual(report["rows"], 0)
        self.assertIn("no prediction rows", report["errors"][0])
