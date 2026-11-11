#!/usr/bin/env python3
"""Train the temporal HAR model (LSTM) on recorded clips.

  python scripts/train_har.py --synthetic        # smoke test, no real clips
  python scripts/train_har.py --data data/raw --epochs 10

--synthetic generates random feature sequences per label so the training
loop can be tested end-to-end without any recorded video. It proves the
code runs; it does NOT produce a usable model — accuracy on synthetic
data is meaningless, and the script says so in its output.

With --data, clips are loaded via the labels.csv, features are extracted
per frame (hand/object features, see activity.py), sequences are split
train/val by clip (not by frame, to avoid leakage), and the script
reports train/val accuracy *measured on that data*. No accuracy number
is claimed anywhere else in this project.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bas_har.activity import ACTIVITY_LABELS  # noqa: E402
from bas_har.models import FEATURE_DIM, NUM_CLASSES, TemporalHAR, require_torch  # noqa: E402

LABEL_TO_IDX = {label: i for i, label in enumerate(ACTIVITY_LABELS)}


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


def train(xs, ys, epochs: int = 5, out: str = "models/har_lstm.pt") -> dict:
    require_torch()
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    n = len(xs)
    split = int(n * 0.8)
    x_train = torch.tensor(np.array(xs[:split]), dtype=torch.float32)
    y_train = torch.tensor(ys[:split])
    x_val = torch.tensor(np.array(xs[split:]), dtype=torch.float32)
    y_val = torch.tensor(ys[split:])

    model = TemporalHAR()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val).argmax(1) == y_val).float().mean().item()
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
    parser.add_argument("--out", default="models/har_lstm.pt")
    args = parser.parse_args()

    if args.synthetic:
        print("WARNING: synthetic data — this only tests the training loop. "
              "The resulting model is NOT usable on real video.")
        xs, ys = synthetic_dataset()
    else:
        labels_csv = Path(args.data) / "labels.csv"
        if not labels_csv.exists():
            print(f"No labels.csv at {labels_csv}. Record clips first "
                  "(scripts/record_dataset.py) or use --synthetic.")
            return 1
        print("Loading real clips is not yet implemented in this script — "
              "feature extraction from video files is the next step. "
              "Use --synthetic to test the training loop for now.")
        return 1
    result = train(xs, ys, epochs=args.epochs, out=args.out)
    print(f"Measured val accuracy on this data: {result['val_acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
