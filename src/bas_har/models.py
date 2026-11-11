"""Learned temporal activity-recognition model (PyTorch, CPU).

A small LSTM over sequences of ``FrameFeatures.to_vector()`` (12 features
per frame). This is the *real-model* path: it only runs when PyTorch is
installed and a trained checkpoint exists.

No accuracy number is claimed anywhere in this project. Accuracy is
"to be measured on your recorded dataset" — see ``scripts/train_har.py``,
which trains on recorded clips (or synthetic data for a smoke test) and
reports train/val accuracy measured on that data.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

FEATURE_DIM = 12
NUM_CLASSES = 10  # 8 protocol steps + IDLE + UNKNOWN


def require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install the full requirements "
            "(pip install -r requirements.txt) or use the rule-based "
            "recogniser in src/bas_har/activity.py, which needs no model."
        )


if HAS_TORCH:

    class TemporalHAR(nn.Module):
        """LSTM classifier: (batch, seq_len, 12) -> (batch, 10) logits."""

        def __init__(
            self,
            feature_dim: int = FEATURE_DIM,
            hidden_dim: int = 64,
            num_layers: int = 1,
            num_classes: int = NUM_CLASSES,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                feature_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1])

else:  # torch absent: define a placeholder that fails with a clear message

    class TemporalHAR:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            require_torch()


def count_parameters(model) -> int:  # type: ignore[no-untyped-def]
    require_torch()
    return sum(p.numel() for p in model.parameters())
