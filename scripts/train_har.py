#!/usr/bin/env python3
"""Train the temporal HAR model (LSTM) on recorded clips.

  python scripts/train_har.py --synthetic        # smoke test, no real clips
  python scripts/train_har.py --data data/raw --epochs 10

--synthetic generates random feature sequences per label so the training
loop can be tested end-to-end without any recorded video. It proves the
code runs; it does NOT produce a usable model — accuracy on synthetic
data is meaningless, and the script says so in its output.

With --data, clips listed in ``data/raw/labels.csv`` (written by
``scripts/record_dataset.py``) are loaded per clip: each video is read
frame by frame, hand/object features are extracted with the same
``activity.extract_features`` / ``vision`` helpers the live system uses
(MediaPipe hands + pose when installed, HSV colour detection for the
red/yellow containers otherwise), and each clip becomes one fixed-length
sequence of ``FrameFeatures.to_vector()`` vectors (12 features/frame).
Sequences are split train/val *by clip, not by frame*, to avoid leakage
(frames from one recording must not appear in both sets). The script
reports train/val accuracy *measured on that data*. No accuracy number
is claimed anywhere else in this project.

Known limits, stated plainly:
- The cardboard box is not colour-detected (only red/yellow containers
  are, via HSV), so ``box`` features are (0, 0) unless a YOLO detector
  is wired in later. Hand + red/yellow motion still carries the signal.
- MediaPipe is optional: without it, hand landmarks are absent and the
  sequence relies on container positions alone. Install the full
  requirements (``pip install -r requirements.txt``) for hand tracking.
- With only a few clips per label the val split is tiny and its accuracy
  is noisy — record 30-50 clips per action (see record_dataset.py) before
  trusting the number.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bas_har.activity import ACTIVITY_LABELS, extract_features  # noqa: E402
from bas_har.models import FEATURE_DIM, NUM_CLASSES, TemporalHAR, require_torch  # noqa: E402

LABEL_TO_IDX = {label: i for i, label in enumerate(ACTIVITY_LABELS)}

# A clip with fewer usable frames than this is skipped, not padded into
# a meaningless sequence.
MIN_FRAMES = 5


def synthetic_dataset(n_per_label: int = 40, seq_len: int = 30, seed: int = 0):
    """Random sequences with a label-specific mean offset, so the model
    can learn *something* — only to prove the training loop works."""
    import numpy as np

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label, idx in LABEL_TO_IDX.items():
        for _ in range(n_per_label):
            seq = rng.normal(loc=idx * 0.1, scale=0.5, size=(seq_len, FEATURE_DIM))
            xs.append(seq)
            ys.append(idx)
    return xs, ys


def clip_to_sequence(path: Path, seq_len: int = 30, tracker=None) -> list | None:
    """Extract one fixed-length feature sequence from a recorded clip.

    Returns ``None`` when the clip cannot be read or has too few frames.
    ``tracker`` is an optional ``vision.PoseHandTracker`` (created once by
    the caller so MediaPipe is not re-initialised per clip); when it is
    ``None`` or unavailable, hand features are absent and the sequence
    still carries the red/yellow container positions.
    """
    from bas_har.vision import detect_coloured_containers, require_cv2

    require_cv2()
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  warning: cannot open {path}, skipping")
        return None
    vectors: list[list[float]] = []
    prev_hand: tuple[float, float] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            objects: dict = {}
            for det in detect_coloured_containers(frame):
                # Keep the largest contour per colour (red is matched by
                # two HSV ranges, so it can appear twice).
                cur = objects.get(det.label)
                if cur is None or det.size > cur[2]:
                    objects[det.label] = (det.x, det.y, det.size)
            hand_landmarks = None
            pose_landmarks = None
            if tracker is not None and tracker.available:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_landmarks, hand = tracker.track(frame_rgb)
                if hand.visible:
                    hand_landmarks = hand.landmarks
            feats = extract_features(
                hand_landmarks=hand_landmarks,
                pose_landmarks=pose_landmarks or None,
                objects=objects,
                prev_hand=prev_hand,
            )
            if feats.hand_visible:
                prev_hand = (feats.hand_x, feats.hand_y)
            vectors.append(feats.to_vector())
    finally:
        cap.release()
    if len(vectors) < MIN_FRAMES:
        print(f"  warning: {path} has only {len(vectors)} frames, skipping")
        return None
    if len(vectors) >= seq_len:
        # Evenly spaced frames across the whole clip.
        idxs = [round(i * (len(vectors) - 1) / (seq_len - 1)) for i in range(seq_len)]
        return [vectors[i] for i in idxs]
    # Short clip: pad by repeating the last frame.
    return vectors + [vectors[-1]] * (seq_len - len(vectors))


def read_labels_csv(data_dir: Path) -> list[tuple[Path, str]]:
    """Read ``labels.csv`` (no header, written by record_dataset.py);
    tolerate a header row. Returns [(clip_path, label), ...]."""
    labels_csv = data_dir / "labels.csv"
    rows: list[tuple[Path, str]] = []
    with labels_csv.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            clip, label = row[0].strip(), row[1].strip()
            if clip == "clip_path" and label == "label":
                continue
            if label not in LABEL_TO_IDX:
                print(f"  warning: unknown label {label!r} for {clip}, skipping")
                continue
            # record_dataset.py writes paths relative to the repo root
            # (e.g. data/raw/OPEN_BOX/clip_....mp4), so try that first,
            # then relative to the data dir itself.
            path = None
            for candidate in (Path(clip), data_dir / clip,
                              data_dir.parent / clip):
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                print(f"  warning: missing clip file {clip}, skipping")
                continue
            rows.append((path, label))
    return rows


def load_real_dataset(data_dir: Path, seq_len: int = 30, seed: int = 0):
    """Load recorded clips into train/val sequences, split by clip.

    Returns ``(x_train, y_train, x_val, y_val, stats)`` where each x is a
    list of sequences shaped (seq_len, FEATURE_DIM) and stats reports the
    clip counts per label and per split.
    """
    from bas_har.vision import PoseHandTracker

    rows = []
    labels_csv = data_dir / "labels.csv"
    if not labels_csv.exists():
        return [], [], [], [], {"clips": {}, "train": 0, "val": 0}
    rows = read_labels_csv(data_dir)
    tracker = PoseHandTracker()
    if not tracker.available:
        print("  note: MediaPipe not available — training on container "
              "positions only (no hand landmarks). "
              "Install the full requirements for hand tracking.")
    clips: list[tuple[list, int]] = []
    per_label: dict[str, int] = {}
    for path, label in rows:
        seq = clip_to_sequence(path, seq_len=seq_len, tracker=tracker)
        if seq is None:
            continue
        clips.append((seq, LABEL_TO_IDX[label]))
        per_label[label] = per_label.get(label, 0) + 1
    if not clips:
        return [], [], [], [], {"clips": per_label, "train": 0, "val": 0}
    rng = random.Random(seed)
    rng.shuffle(clips)
    n_val = max(1, int(round(len(clips) * 0.2))) if len(clips) >= 2 else 0
    val = clips[:n_val]
    train = clips[n_val:]
    if not train:  # a single clip cannot be split; train on it, no val
        train, val = clips, []
    x_train = [s for s, _ in train]
    y_train = [y for _, y in train]
    x_val = [s for s, _ in val]
    y_val = [y for _, y in val]
    stats = {"clips": per_label, "train": len(train), "val": len(val)}
    return x_train, y_train, x_val, y_val, stats


def train(xs, ys, epochs: int = 5, out: str = "models/har_lstm.pt",
          x_val=None, y_val=None) -> dict:
    require_torch()
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    if x_val is None or y_val is None:
        # Legacy path (synthetic smoke test): split by sequence.
        n = len(xs)
        split = int(n * 0.8)
        x_train = torch.tensor(np.array(xs[:split]), dtype=torch.float32)
        y_train = torch.tensor(ys[:split])
        x_val_t = torch.tensor(np.array(xs[split:]), dtype=torch.float32)
        y_val_t = torch.tensor(ys[split:])
    else:
        x_train = torch.tensor(np.array(xs), dtype=torch.float32)
        y_train = torch.tensor(ys)
        x_val_t = torch.tensor(np.array(x_val), dtype=torch.float32)
        y_val_t = torch.tensor(y_val)
    model = TemporalHAR()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val_t).argmax(1) == y_val_t).float().mean().item()
        print(f"epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}  val_acc={val_acc:.3f}")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to {out_path}")
    return {"val_acc": val_acc, "checkpoint": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the HAR LSTM")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/har_lstm.pt")
    args = parser.parse_args()

    if args.synthetic:
        print("WARNING: synthetic data — this only tests the training loop. "
              "The resulting model is NOT usable on real video.")
        xs, ys = synthetic_dataset()
        result = train(xs, ys, epochs=args.epochs, out=args.out)
        print(f"Measured val accuracy on this data: {result['val_acc']:.3f}")
        return 0

    data_dir = Path(args.data)
    labels_csv = data_dir / "labels.csv"
    if not labels_csv.exists():
        print(f"No labels.csv at {labels_csv}. Record clips first "
              "(scripts/record_dataset.py) or use --synthetic.")
        return 1
    x_train, y_train, x_val, y_val, stats = load_real_dataset(
        data_dir, seq_len=args.seq_len, seed=args.seed)
    if not x_train:
        print("No usable clips found. Record clips first "
              "(scripts/record_dataset.py) or use --synthetic.")
        return 1
    print(f"Clips per label: {stats['clips']}")
    print(f"Train sequences: {len(x_train)}  Val sequences: {len(x_val)} "
          f"(split by clip, seq_len={args.seq_len})")
    kwargs = {"x_val": x_val, "y_val": y_val} if x_val else {}
    result = train(x_train, y_train, epochs=args.epochs, out=args.out, **kwargs)
    print(f"Measured val accuracy on this data: {result['val_acc']:.3f}")
    if len(x_val) < 5:
        print("Note: val set is very small — this accuracy is noisy. "
              "Record more clips per label before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
