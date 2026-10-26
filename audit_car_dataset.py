"""Read-only UCI Car Evaluation dataset schema and coverage audit.

This validates source data; it does not train or evaluate any model.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FEATURE_VALUES = {
    "buying": ("vhigh", "high", "med", "low"),
    "maint": ("vhigh", "high", "med", "low"),
    "doors": ("2", "3", "4", "5more"),
    "persons": ("2", "4", "more"),
    "lug_boot": ("small", "med", "big"),
    "safety": ("low", "med", "high"),
}
CLASS_VALUES = ("unacc", "acc", "good", "vgood")


def audit_dataset(path, *, require_complete=False):
    """Validate row width, allowed values and uniqueness of six-feature tuples."""
    combinations = Counter()
    classes = Counter()
    issues = []
    row_count = 0
    with open(path, newline="", encoding="utf-8-sig") as source:
        for line_number, row in enumerate(csv.reader(source), 1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            row_count += 1
            if len(row) != 7:
                issues.append(f"line {line_number}: expected 7 columns, got {len(row)}")
                continue
            row = [value.strip() for value in row]
            bad_fields = [name for name, value in zip(FEATURE_VALUES, row[:6]) if value not in FEATURE_VALUES[name]]
            if row[6] not in CLASS_VALUES:
                bad_fields.append("class")
            if bad_fields:
                issues.append(f"line {line_number}: invalid value in {', '.join(bad_fields)}")
                continue
            key = tuple(row[:6])
            combinations[key] += 1
            classes[row[6]] += 1
            if combinations[key] > 1:
                issues.append(f"line {line_number}: duplicate feature combination")
    expected_count = 1
    for values in FEATURE_VALUES.values():
        expected_count *= len(values)
    missing_count = expected_count - len(combinations)
    if require_complete and missing_count:
        issues.append(f"missing {missing_count} of {expected_count} possible feature combinations")
    return {
        "valid": not issues,
        "rows": row_count,
        "unique_combinations": len(combinations),
        "expected_combinations": expected_count,
        "missing_combinations": missing_count,
        "class_counts": {name: classes[name] for name in CLASS_VALUES},
        "issue_count": len(issues),
        "issues": issues[:10],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate the UCI Car Evaluation dataset without training")
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).with_name("car.data"))
    parser.add_argument("--require-complete", action="store_true", help="require all 1728 feature combinations")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.path, require_complete=args.require_complete)
    except (OSError, UnicodeError, csv.Error) as exc:
        parser.error(f"cannot read dataset: {exc}")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Rows: {report['rows']} | unique combinations: {report['unique_combinations']}/{report['expected_combinations']}")
        print(f"Status: {'PASS' if report['valid'] else 'FAIL'} | issues: {report['issue_count']}")
        for issue in report["issues"]:
            print(f"- {issue}")
        if report["issue_count"] > len(report["issues"]):
            print(f"... {report['issue_count'] - len(report['issues'])} additional issues omitted")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
