"""Instrument detection pipeline.

Two modes:
1. **YOLOv8** (legacy) — generic object detection with bounding boxes.
2. **ResNet50 tool classifier** (Cholec80-trained) — multi-label surgical
   instrument presence classification fine-tuned on Cholec80 tool
   annotations.  No bounding boxes, but proper surgical instrument labels.

Both pipelines output the same JSONL format so downstream scene assembly
works unchanged.

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


# ── ResNet50 tool classifier pipeline ────────────────────────────────

class ToolClassifierPipeline:
    """Multi-label surgical instrument presence classifier.

    Uses a ResNet50 fine-tuned on Cholec80 tool annotations to predict
    which of the 7 surgical instruments are present in each frame.
    Outputs the same JSONL format as :class:`DetectionPipeline` for
    backward compatibility with scene assembly.

    Supports TensorRT-compiled models for accelerated inference
    (``trt_path`` parameter).

    Parameters
    ----------
    model_weights : str | Path
        Path to a trained ``.pt`` checkpoint from ``train_tool.py``.
    trt_path : str | Path | None
        Path to a TensorRT-compiled model from ``export_tensorrt.py``.
        When provided, this is used instead of the PyTorch model for
        significantly faster inference (~1,300 FPS on RTX 5060 Ti).
    device : str | None
        PyTorch device string.  ``None`` auto-selects CUDA.
    threshold : float
        Sigmoid threshold for tool presence (default 0.5).
    half : bool
        Use FP16 inference.
    batch_size : int
        Frames per forward pass.
    img_size : int
        Input image size (centre-cropped square).
    """

    def __init__(
        self,
        model_weights: str | Path,
        trt_path: str | Path | None = None,
        device: str | None = None,
        threshold: float = 0.5,
        half: bool = True,
        batch_size: int = 32,
        img_size: int = 224,
    ) -> None:
        self.threshold = threshold
        self.half = half
        self.batch_size = batch_size
        self.img_size = img_size
        self.using_trt = False

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.half and self.device == "cpu":
            log.warning("FP16 not supported on CPU – falling back to FP32.")
            self.half = False

        # Try TensorRT first
        if trt_path and Path(trt_path).exists() and self.device != "cpu":
            try:
                import torch_tensorrt  # noqa: F401
                log.info("Loading TensorRT tool classifier: %s", trt_path)
                self.model = torch.export.load(str(trt_path)).module()
                self.model = self.model.to(self.device)
                self.model.eval()
                self.using_trt = True
                log.info("TensorRT tool classifier loaded (FP16 accelerated)")
            except Exception as e:
                log.warning("TensorRT load failed (%s), falling back to PyTorch", e)
                self.using_trt = False

        # Fallback: standard PyTorch model
        if not self.using_trt:
            from torchvision import models
            from torchvision.models import ResNet50_Weights

            log.info("Loading tool classifier: %s  (device=%s)", model_weights, self.device)
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(in_features, len(CHOLEC80_INSTRUMENTS))

            state = torch.load(str(model_weights), map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            backbone.load_state_dict(state)

            self.model = backbone.to(self.device)
            self.model.eval()
            if self.half:
                self.model.half()

        mode = "TensorRT FP16" if self.using_trt else "PyTorch"
        log.info("Tool classifier ready [%s] – %d instruments, threshold=%.2f",
                 mode, len(CHOLEC80_INSTRUMENTS), self.threshold)

    def _preprocess(self, bgr_frames: list[np.ndarray]) -> torch.Tensor:
        """BGR frames -> normalised (N, 3, img_size, img_size) tensor."""
        import cv2 as _cv2
        tensors = []
        for bgr in bgr_frames:
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = (self.img_size + 32) / min(h, w)
            rgb = _cv2.resize(rgb, (int(w * scale), int(h * scale)))
            nh, nw = rgb.shape[:2]
            y0, x0 = (nh - self.img_size) // 2, (nw - self.img_size) // 2
            rgb = rgb[y0:y0 + self.img_size, x0:x0 + self.img_size]
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensors.append(t)
        batch = torch.stack(tensors).to(self.device)
        if self.half:
            batch = batch.half()
        return batch

    @torch.inference_mode()
    def _predict_batch(self, bgr_frames: list[np.ndarray]) -> list[list[Detection]]:
        """Classify tool presence for a batch of frames."""
        batch = self._preprocess(bgr_frames)
        logits = self.model(batch)
        probs = torch.sigmoid(logits.float())  # (N, 7)

        results = []
        for i in range(probs.size(0)):
            dets = []
            for tool_id in range(len(CHOLEC80_INSTRUMENTS)):
                conf = probs[i, tool_id].item()
                if conf >= self.threshold:
                    dets.append(Detection(
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        class_id=tool_id,
                        class_name=CHOLEC80_INSTRUMENTS[tool_id],
                        confidence=conf,
                    ))
            results.append(dets)
        return results

    def process_video(
        self,
        video_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> Iterator[FrameResult]:
        """Run tool classification on every (strided) frame."""
        video_path = Path(video_path)
        if video_id is None:
            video_id = video_path.stem

        with VideoReader(video_path, stride=stride) as reader:
            log.info(
                "Tool-classifying %s  (%dx%d, %.1f fps, ~%d frames, stride=%d)",
                video_id, reader.width, reader.height,
                reader.fps, reader.frame_count, stride,
            )
            t0 = time.perf_counter()
            frames_processed = 0

            for batch in reader.batched_frames(self.batch_size):
                indices, timestamps, bgr_frames = zip(*batch)
                dets_per_frame = self._predict_batch(list(bgr_frames))

                for idx, ts, dets in zip(indices, timestamps, dets_per_frame):
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

    def process_video_to_jsonl(
        self,
        video_path: Path,
        output_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> dict:
        """Process a video and write results to JSONL."""
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
