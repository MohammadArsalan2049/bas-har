"""Local video recording and streaming.

Recording and streaming run *independently* of the inference pipeline:
a network problem must never stop on-board monitoring. Both helpers fail
gracefully — recording without OpenCV, or streaming without ffmpeg, returns
a clear reason instead of crashing the experiment.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .vision import HAS_CV2, require_cv2


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class LocalRecorder:
    """Saves the original camera feed to disk with OpenCV VideoWriter."""

    def __init__(self, output_path: str | Path, fps: float = 20.0,
                 width: int = 1280, height: int = 720) -> None:
        require_cv2()
        import cv2

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        self.recording = True

    def write(self, frame) -> None:
        if self.recording:
            self.writer.write(frame)

    def stop(self) -> Path:
        self.recording = False
        self.writer.release()
        return self.output_path


def build_ffmpeg_rtsp_command(
    input_source: str = "0", rtsp_url: str = "rtsp://127.0.0.1:8554/bas",
    fps: int = 20,
) -> list[str]:
    """Return the ffmpeg command that would stream the feed to an RTSP
    endpoint. Does not run it — see ``start_rtsp_stream``."""
    return [
        "ffmpeg", "-f", "v4l2" if input_source.isdigit() else "lavfi",
        "-i", input_source if input_source.isdigit() else input_source,
        "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
        "-f", "rtsp", rtsp_url,
    ]


def start_rtsp_stream(
    input_source: str = "0", rtsp_url: str = "rtsp://127.0.0.1:8554/bas",
    fps: int = 20,
) -> subprocess.Popen:
    """Start ffmpeg streaming in a subprocess. Raises a clear error if
    ffmpeg is missing."""
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed, so RTSP streaming is unavailable. "
            "Local recording and inference keep working without it."
        )
    cmd = build_ffmpeg_rtsp_command(input_source, rtsp_url, fps)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stream_info(rtsp_url: str = "rtsp://127.0.0.1:8554/bas") -> dict:
    return {
        "rtsp_url": rtsp_url,
        "ffmpeg_available": ffmpeg_available(),
        "opencv_available": HAS_CV2,
        "note": "Streaming runs independently of inference; "
                "a network problem does not stop on-board monitoring.",
    }
