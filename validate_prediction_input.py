"""Read-only validation of batch prediction inputs (no model or dependencies needed)."""

import argparse
import csv
import json
from pathlib import Path

FEATURE_VALUES = {
    "buying": {"vhigh", "high", "med", "low"},
    "maint": {"vhigh", "high", "med", "low"},
    "doors": {"2", "3", "4", "5more"},
    "persons": {"2", "4", "more"},
    "lug_boot": {"small", "med", "big"},
    "safety": {"low", "med", "high"},
}
FEATURES = set(FEATURE_VALUES)


def _validate_record(record, location):
    if not isinstance(record, dict):
        return [f"{location}: expected an object with six features"]
    errors = []
    missing = sorted(FEATURES - record.keys())
    extra = sorted(record.keys() - FEATURES)
    if missing:
        errors.append(f"{location}: missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"{location}: unexpected columns: {', '.join(map(str, extra))}")
    for feature in FEATURE_VALUES:
        if feature not in record:
            continue
        value = record[feature]
        if not isinstance(value, str) or value not in FEATURE_VALUES[feature]:
            errors.append(
                f"{location}: invalid {feature}={value!r}; "
                f"allowed: {', '.join(sorted(FEATURE_VALUES[feature]))}"
            )
    return errors


def validate_file(path):
    """Return a JSON-serializable report without modifying the input file."""
    path = Path(path)
    report = {"file": str(path), "rows": 0, "valid_rows": 0, "errors": []}
    if path.suffix.lower() not in {".csv", ".json"}:
        report["errors"].append("Unsupported extension: use .csv or .json")
        return report

    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                headers = reader.fieldnames
                if not headers:
                    report["errors"].append("CSV is empty or missing a header")
                    return report
                if len(headers) != len(set(headers)):
                    report["errors"].append("CSV has duplicate column names")
                    return report
                for row in reader:
                    report["rows"] += 1
                    location = f"CSV line {reader.line_num}"
                    # DictReader stores surplus fields under the None key.
                    if None in row:
                        report["errors"].append(f"{location}: too many fields")
                        continue
                    errors = _validate_record(row, location)
                    report["errors"].extend(errors)
                    if not errors:
                        report["valid_rows"] += 1
        else:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                report["errors"].append("JSON root must be an array of objects")
                return report
            for number, row in enumerate(data, 1):
                report["rows"] += 1
                errors = _validate_record(row, f"JSON item {number}")
                report["errors"].extend(errors)
                if not errors:
                    report["valid_rows"] += 1
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        report["errors"].append(f"Cannot read input: {exc}")

    if report["rows"] == 0 and not report["errors"]:
        report["errors"].append("Input contains no prediction rows")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate Car Evaluation batch inputs offline")
    parser.add_argument("file", help="CSV or JSON file with six categorical features")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    report = validate_file(args.file)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Rows: {report['rows']}; valid: {report['valid_rows']}; "
              f"errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"  - {error}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
