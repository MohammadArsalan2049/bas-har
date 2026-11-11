"""BAS-HAR using AI — Human Activity Recognition for on-board BAS experiments.

Independent project. Offline-first: no cloud services, no LLM, no internet
dependency at runtime. Demo mode runs with no camera, no GPU and no heavy
ML libraries installed.
"""

__version__ = "0.1.0"
__title__ = "BAS-HAR using AI"

from .protocol import ProtocolEngine, TemporalConfirmer  # noqa: F401
