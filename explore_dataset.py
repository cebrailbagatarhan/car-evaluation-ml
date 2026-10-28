"""Explore the UCI Car Evaluation dataset without training a model."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

COLUMNS = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")


def summarize(path: Path, filters: dict[str, str], *, include_percentages: bool = False) -> dict[str, object]:
    counts: Counter[str] = Counter()
    matched = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != len(COLUMNS):
                raise ValueError(f"Invalid column count on line {line_number}: expected 7, got {len(row)}")
            record = dict(zip(COLUMNS, row))
            if all(record[key] == value for key, value in filters.items()):
                counts[record["class"]] += 1
                matched += 1
    summary = {"matched": matched, "filters": filters, "classes": dict(sorted(counts.items()))}
    if include_percentages:
        summary["class_percentages"] = {
            label: round(count * 100 / matched, 1) for label, count in sorted(counts.items())
        } if matched else {}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore Car Evaluation class distributions")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("car.data"))
    parser.add_argument("--buying", choices=("vhigh", "high", "med", "low"))
    parser.add_argument("--persons", choices=("2", "4", "more"))
    parser.add_argument("--safety", choices=("low", "med", "high"))
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--percent", action="store_true", help="include per-class percentages of matching rows")
    args = parser.parse_args(argv)
    filters = {key: value for key in ("buying", "persons", "safety") if (value := getattr(args, key)) is not None}
    try:
        summary = summarize(args.data, filters, include_percentages=args.percent)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot read dataset: {exc}\n")
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Matching cars: {summary['matched']}")
        for label, count in summary["classes"].items():
            percentage = f" ({summary['class_percentages'][label]:.1f}%)" if args.percent else ""
            print(f"  {label}: {count}{percentage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
