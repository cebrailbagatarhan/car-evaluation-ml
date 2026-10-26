import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from audit_car_dataset import audit_dataset, main


class AuditDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "car.data"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_valid_subset_without_complete_requirement(self):
        self.write("low,low,4,4,big,high,good\n")
        result = audit_dataset(self.path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["class_counts"]["good"], 1)

    def test_complete_requirement_flags_missing_combinations(self):
        self.write("low,low,4,4,big,high,good\n")
        result = audit_dataset(self.path, require_complete=True)
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_combinations"], 1727)

    def test_duplicate_is_rejected(self):
        self.write("low,low,4,4,big,high,good\nlow,low,4,4,big,high,good\n")
        self.assertIn("duplicate", audit_dataset(self.path)["issues"][0])

    def test_bad_values_and_column_count(self):
        self.write("low,low,4,4,big,unsafe,good\nlow,low\n")
        report = audit_dataset(self.path)
        self.assertEqual(report["issue_count"], 2)
        self.assertEqual(report["unique_combinations"], 0)

    def test_json_cli_exit_code(self):
        self.write("low,low,4,4,big,high,good\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(self.path), "--json"]), 0)
        self.assertTrue(json.loads(output.getvalue())["valid"])


if __name__ == "__main__":
    unittest.main()
