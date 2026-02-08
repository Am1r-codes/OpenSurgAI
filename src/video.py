"""Shared video reading utilities.

Provides a memory-efficient :class:`VideoReader` backed by OpenCV that
yields frames individually or in batches.  Used by both the detection
and segmentation pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Memory-efficient video reader backed by OpenCV.

    Yields ``(frame_idx, timestamp_sec, bgr_frame)`` tuples.  Supports
    an optional *stride* to skip frames (e.g. ``stride=2`` processes
    every other frame).
    """

    def __init__(self, video_path: Path, stride: int = 1) -> None:
        self.video_path = Path(video_path)
        self.stride = max(1, stride)
        self._cap: cv2.VideoCapture | None = None

    # ── properties (available after open) ─────────────────────────────

    @property
    def fps(self) -> float:
        assert self._cap is not None, "call open() first"
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        assert self._cap is not None, "call open() first"
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        assert self._cap is not None, "call open() first"
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        assert self._cap is not None, "call open() first"
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── context manager ───────────────────────────────────────────────

    def open(self) -> VideoReader:
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ── iteration ─────────────────────────────────────────────────────

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_idx, timestamp_sec, bgr_frame)``."""
        assert self._cap is not None, "call open() first"
        fps = self.fps
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % self.stride == 0:
                yield idx, idx / fps if fps > 0 else 0.0, frame
            idx += 1

    def batched_frames(
        self, batch_size: int = 16
    ) -> Iterator[list[tuple[int, float, np.ndarray]]]:
        """Yield lists of up to *batch_size* ``(idx, ts, frame)`` tuples."""
        batch: list[tuple[int, float, np.ndarray]] = []
        for item in self.frames():
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
