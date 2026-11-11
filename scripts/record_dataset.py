#!/usr/bin/env python3
"""Record webcam clips per activity label for the training dataset.

How to use it (do this on a laptop with a webcam):
  1. Build a miniature experiment station: a cardboard box, a red
     container and a yellow container, on a table in front of the camera.
  2. Run:  python scripts/record_dataset.py --label OPEN_BOX --clips 5 --seconds 4
  3. Repeat for each label in src/bas_har/activity.py (ACTIVITY_LABELS).
     Record each action several times, from different angles, at different
     speeds, with different people and with partial occlusion.
  4. Clips are saved to data/raw/<label>/clip_YYYYMMDD_HHMMSS.mp4 and
     listed in data/raw/labels.csv.

Aim for 30-50 clips per action as a starting target, then expand based
on measured performance. Keep recordings from different sessions or
participants in separate train/val/test splits to avoid data leakage.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bas_har.activity import ACTIVITY_LABELS  # noqa: E402
from bas_har.vision import VideoCapture, require_cv2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record dataset clips")
    parser.add_argument("--label", required=True, choices=ACTIVITY_LABELS)
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    require_cv2()
    import cv2

    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = Path(args.out) / "labels.csv"
    cap = VideoCapture(args.camera)
    try:
        for i in range(args.clips):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"clip_{stamp}_{i}.mp4"
            print(f"Recording clip {i + 1}/{args.clips} for {args.label} "
                  f"({args.seconds}s) -> {path}")
            print("  Perform the action now. Press Ctrl-C to stop early.")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     20.0, (1280, 720))
            start = cv2.getTickCount()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                cv2.imshow("Recording — press q to stop this clip", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
                if elapsed >= args.seconds:
                    break
            writer.release()
            with labels_csv.open("a", newline="") as f:
                csv.writer(f).writerow([str(path), args.label])
            print(f"  Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
