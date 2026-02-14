"""Video frame extraction for Nemotron VL visual analysis.

Extracts individual frames from surgical videos and encodes them as
base64 JPEG for sending to the Nemotron VL NIM API.  Uses OpenCV (cv2)
which is already installed in the opensurgai environment.

Usage::

    from src.explanation.frame_extractor import extract_frame_at_time

    b64 = extract_frame_at_time("data/cholec80/videos/video49.mp4", 120.5)
    # b64 is a base64-encoded JPEG string ready for Nemotron VL
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def extract_frame_at_time(
    video_path: str | Path,
    time_sec: float,
    max_width: int = 1024,
    jpeg_quality: int = 85,
) -> str | None:
    """Extract a single frame from a video at the given timestamp.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    time_sec : float
        Timestamp in seconds to extract the frame from.
    max_width : int
        Maximum width for the output image (aspect ratio preserved).
        Smaller images = faster VLM API calls.
    jpeg_quality : int
        JPEG compression quality (1-100).

    Returns
    -------
    str or None
        Base64-encoded JPEG string, or None if extraction fails.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Clamp time to valid range
        time_sec = max(0.0, min(time_sec, duration - 1 / fps))
        frame_num = int(time_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            log.error("Failed to read frame %d at %.1fs", frame_num, time_sec)
            return None

        # Resize if wider than max_width (preserve aspect ratio)
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            log.error("JPEG encoding failed")
            return None

        b64 = base64.b64encode(buffer).decode("ascii")
        log.info(
            "Extracted frame %d (%.1fs) from %s: %dx%d, %d KB",
            frame_num, time_sec, Path(video_path).name,
            frame.shape[1], frame.shape[0], len(b64) // 1024,
        )
        return b64

    finally:
        cap.release()


def extract_keyframes(
    video_path: str | Path,
    timestamps: list[float],
    max_width: int = 768,
    jpeg_quality: int = 80,
) -> list[dict]:
    """Extract multiple keyframes from a video.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    timestamps : list of float
        List of timestamps in seconds.
    max_width : int
        Maximum width for output images.
    jpeg_quality : int
        JPEG compression quality.

    Returns
    -------
    list of dict
        Each dict has: time_sec, frame_b64, width, height.
        Entries where extraction failed are omitted.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    results = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        for ts in sorted(timestamps):
            ts_clamped = max(0.0, min(ts, duration - 1 / fps))
            frame_num = int(ts_clamped * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(
                    frame,
                    (max_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            results.append({
                "time_sec": ts,
                "frame_b64": base64.b64encode(buffer).decode("ascii"),
                "width": frame.shape[1],
                "height": frame.shape[0],
            })

        log.info(
            "Extracted %d/%d keyframes from %s",
            len(results), len(timestamps), Path(video_path).name,
        )
    finally:
        cap.release()

    return results


def get_video_info(video_path: str | Path) -> dict | None:
    """Get basic video metadata.

    Returns dict with: fps, total_frames, duration_sec, width, height.
    Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": total_frames / fps,
            "width": w,
            "height": h,
        }
    finally:
        cap.release()
