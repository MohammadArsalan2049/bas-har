"""Tests for the real-clip loader in scripts/train_har.py.

Generates small synthetic mp4s (a red rectangle moving across a white
background) with OpenCV, so no webcam or recorded dataset is needed.
Skipped when OpenCV is absent. Torch is not needed — these tests only
cover loading / feature extraction, not the training loop.
"""
import csv
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_har import FEATURE_DIM, clip_to_sequence, load_real_dataset  # noqa: E402


def _make_clip(path: Path, frames: int = 40, size=(320, 240)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             20.0, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        # Red rectangle (BGR 0,0,255) drifting right; yellow (0,255,255) static.
        x = 20 + i * 3
        cv2.rectangle(frame, (x, 100), (x + 60, 160), (0, 0, 255), -1)
        cv2.rectangle(frame, (200, 100), (260, 160), (0, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _dataset(tmp_path: Path, n_clips: int = 4) -> Path:
    data_dir = tmp_path / "data" / "raw"
    rows = []
    for i in range(n_clips):
        clip = data_dir / "PICK_RED" / f"clip_{i}.mp4"
        _make_clip(clip)
        rows.append([str(clip), "PICK_RED"])
    with (data_dir / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return data_dir


def test_clip_to_sequence_shape_and_red_detected(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    seq = clip_to_sequence(clip, seq_len=30)
    assert seq is not None
    assert len(seq) == 30
    assert all(len(v) == FEATURE_DIM for v in seq)
    # to_vector layout: [hand(6), box x/y, red x/y, yellow x/y]
    red_xs = [v[8] for v in seq]
    assert any(x > 0 for x in red_xs), "red container should be detected"
    yellow_xs = [v[10] for v in seq]
    assert any(x > 0 for x in yellow_xs), "yellow container should be detected"


def test_clip_too_short_is_skipped(tmp_path):
    clip = tmp_path / "short.mp4"
    _make_clip(clip, frames=2)
    assert clip_to_sequence(clip, seq_len=30) is None


def test_load_real_dataset_split_by_clip(tmp_path):
    data_dir = _dataset(tmp_path, n_clips=4)
    x_train, y_train, x_val, y_val, stats = load_real_dataset(data_dir, seq_len=30)
    assert stats["clips"] == {"PICK_RED": 4}
    assert len(x_train) + len(x_val) == 4
    assert len(x_train) == 3 and len(x_val) == 1
    assert all(y == 1 for y in y_train + y_val)  # PICK_RED is index 1
    assert all(len(s) == 30 and len(s[0]) == FEATURE_DIM for s in x_train + x_val)


def test_load_real_dataset_missing_csv(tmp_path):
    x_train, y_train, x_val, y_val, stats = load_real_dataset(tmp_path / "empty")
    assert x_train == [] and stats["train"] == 0
