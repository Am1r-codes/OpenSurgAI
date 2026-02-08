"""Instrument detection pipeline using YOLOv8.

Loads a pretrained YOLOv8 model, runs GPU inference on surgical videos,
and outputs per-frame instrument bounding boxes with class labels and
confidence scores.  Designed for ≥30 FPS throughput on RTX 5060 Ti.

Output format (JSON Lines – one JSON object per line):
    {
      "video_id": "video01",
      "frame_idx": 0,
      "timestamp_sec": 0.0,
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "class_id": 0,
          "class_name": "Grasper",
          "confidence": 0.93
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from ultralytics import YOLO

from src.video import VideoReader

log = logging.getLogger(__name__)

# ── Cholec80 surgical instrument classes ──────────────────────────────
# These are the 7 standard Cholec80 tool classes.  When fine-tuning a
# YOLO model on Cholec80, map the model's class indices to these names.
# For a generic pretrained COCO model the YOLO class names are used
# directly (useful for initial smoke-testing before fine-tuning).
CHOLEC80_INSTRUMENTS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single bounding-box detection."""
    bbox: list[float]          # [x1, y1, x2, y2] in pixel coords
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 2) for v in self.bbox],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class FrameResult:
    """Detection results for a single frame."""
    video_id: str
    frame_idx: int
    timestamp_sec: float
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Detection pipeline ────────────────────────────────────────────────

class DetectionPipeline:
    """YOLOv8-based surgical instrument detection pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to a YOLO ``.pt`` weights file **or** one of the standard
        Ultralytics model names (``"yolov8n.pt"``, ``"yolov8s.pt"``, …).
        For Cholec80 use a model fine-tuned on surgical instruments.
    device : str | None
        PyTorch device string.  ``None`` auto-selects CUDA if available.
    conf_threshold : float
        Minimum confidence for a detection to be emitted.
    iou_threshold : float
        IoU threshold for non-maximum suppression.
    img_size : int
        Input image size (pixels) passed to YOLO inference.
    half : bool
        Use FP16 inference for higher throughput on supported GPUs.
    batch_size : int
        Number of frames per inference batch.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8s.pt",
        device: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        half: bool = True,
        batch_size: int = 16,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half = half
        self.batch_size = batch_size

        # ── device selection ──────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.half and self.device == "cpu":
            log.warning("FP16 not supported on CPU – falling back to FP32.")
            self.half = False

        # ── model loading ─────────────────────────────────────────────
        log.info("Loading YOLO model: %s  (device=%s, half=%s)", model_path, self.device, self.half)
        self.model = YOLO(str(model_path))
        # Fuse model layers for faster inference
        self.model.fuse()
        log.info(
            "Model loaded – %d classes: %s",
            len(self.model.names),
            list(self.model.names.values())[:10],
        )

    # ── single-batch inference ────────────────────────────────────────

    def _predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run inference on a batch of BGR frames.

        Returns a list (one per frame) of Detection lists.
        """
        results = self.model.predict(
            source=frames,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            half=self.half,
            verbose=False,
        )
        batch_detections: list[list[Detection]] = []
        for result in results:
            frame_dets: list[Detection] = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    frame_dets.append(Detection(
                        bbox=xyxy,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    ))
            batch_detections.append(frame_dets)
        return batch_detections

    # ── full-video inference ──────────────────────────────────────────

    def process_video(
        self,
        video_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> Iterator[FrameResult]:
        """Run detection on every (strided) frame of a video.

        Yields :class:`FrameResult` objects in frame order.
        """
        video_path = Path(video_path)
        if video_id is None:
            video_id = video_path.stem

        with VideoReader(video_path, stride=stride) as reader:
            log.info(
                "Processing %s  (%dx%d, %.1f fps, ~%d frames, stride=%d)",
                video_id, reader.width, reader.height,
                reader.fps, reader.frame_count, stride,
            )
            t0 = time.perf_counter()
            frames_processed = 0

            for batch in reader.batched_frames(self.batch_size):
                indices, timestamps, bgr_frames = zip(*batch)
                detections_per_frame = self._predict_batch(list(bgr_frames))

                for idx, ts, dets in zip(indices, timestamps, detections_per_frame):
                    frames_processed += 1
                    yield FrameResult(
                        video_id=video_id,
                        frame_idx=idx,
                        timestamp_sec=ts,
                        detections=dets,
                    )

            elapsed = time.perf_counter() - t0
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            log.info(
                "Finished %s – %d frames in %.1fs  (%.1f FPS)",
                video_id, frames_processed, elapsed, fps_actual,
            )

    # ── convenience: save to JSONL ────────────────────────────────────

    def process_video_to_jsonl(
        self,
        video_path: Path,
        output_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> dict:
        """Process a video and write results to a JSON Lines file.

        Returns a summary dict with frame count, detection count, and
        throughput FPS.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        total_detections = 0
        t0 = time.perf_counter()

        with open(output_path, "w") as fh:
            for result in self.process_video(video_path, video_id, stride):
                fh.write(result.to_json() + "\n")
                total_frames += 1
                total_detections += len(result.detections)

        elapsed = time.perf_counter() - t0
        summary = {
            "video_id": video_id or Path(video_path).stem,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "elapsed_sec": round(elapsed, 2),
            "throughput_fps": round(total_frames / elapsed, 1) if elapsed > 0 else 0,
        }
        log.info("Saved %s  (%s)", output_path, summary)
        return summary
