#!/usr/bin/env python3
"""Simple CSV annotation helper for recorded clips.

Usage:
  python scripts/annotate.py --list                 # show unlabelled clips
  python scripts/annotate.py --file data/raw/labels.csv --label OPEN_BOX --clip data/raw/OPEN_BOX/clip_....mp4

Reads/writes data/raw/labels.csv with columns: clip_path, label.
No GUI, no cloud service — edit the CSV directly if you prefer.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABELS_CSV = Path("data/raw/labels.csv")


def read_labels() -> list[dict]:
    if not LABELS_CSV.exists():
        return []
    with LABELS_CSV.open(newline="") as f:
        return list(csv.DictReader(f, fieldnames=["clip_path", "label"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate recorded clips")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--file", default="data/raw/labels.csv")
    parser.add_argument("--label")
    parser.add_argument("--clip")
    args = parser.parse_args()

    if args.list:
        for row in read_labels():
            print(f"{row['clip_path']:60s}  {row['label']}")
        print(f"\n{len(read_labels())} labelled clips in {LABELS_CSV}")
        return 0
    if args.label and args.clip:
        path = Path(args.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([args.clip, args.label])
        print(f"Labelled {args.clip} as {args.label}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
