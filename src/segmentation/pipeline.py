"""Anatomy segmentation pipeline for surgical videos.

Loads a pretrained semantic segmentation model (DeepLabV3 with ResNet-101
backbone by default, via torchvision), runs GPU inference on Cholec80
video frames, and outputs per-frame segmentation masks.

Mask format
-----------
Each mask is a single-channel **uint8 PNG** where each pixel value is
the predicted class index (0 = background).  This encoding is compact
(~10-30 KB per frame with PNG compression), lossless, and directly
overlay-ready — downstream visualisation modules can map class indices
to colours via the ``CLASS_PALETTE``.

Per-frame metadata is written as JSON Lines alongside the masks::

    {
      "video_id": "video01",
      "frame_idx": 0,
      "timestamp_sec": 0.0,
      "mask_file": "video01/000000.png",
      "class_pixel_counts": {"0": 245120, "15": 62480}
    }

Model options
-------------
- ``"deeplabv3_resnet101"`` (default) — best accuracy, ~15-25 FPS at
  512 px on RTX 5060 Ti with FP16.
- ``"deeplabv3_resnet50"`` — faster, slightly less accurate.
- ``"deeplabv3_mobilenet_v3_large"`` — lightest, highest FPS.
- Any custom ``.pt`` checkpoint that exposes a ``state_dict`` with a
  matching architecture.

For surgical-specific anatomy classes, fine-tune one of the above on a
labelled surgical segmentation dataset (e.g., CholecSeg8k) and pass the
checkpoint path via ``--model-weights``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)
from torchvision.transforms import functional as TF

from src.video import VideoReader

log = logging.getLogger(__name__)

# ── Pretrained model registry ─────────────────────────────────────────
# Maps short name → (torchvision constructor, default weights enum).
_MODEL_REGISTRY: dict[str, tuple] = {
    "deeplabv3_resnet101": (
        models.segmentation.deeplabv3_resnet101,
        DeepLabV3_ResNet101_Weights.DEFAULT,
    ),
    "deeplabv3_resnet50": (
        models.segmentation.deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights.DEFAULT,
    ),
    "deeplabv3_mobilenet_v3_large": (
        models.segmentation.deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
}

# ── Default COCO-pretrained class names (21 classes) ──────────────────
COCO_VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ── Colour palette for overlay visualisation (RGB) ────────────────────
# 21 distinct colours for COCO-VOC classes; index 0 is transparent/black.
CLASS_PALETTE = np.array([
    [0,   0,   0],    # 0  background
    [128, 0,   0],    # 1  aeroplane
    [0,   128, 0],    # 2  bicycle
    [128, 128, 0],    # 3  bird
    [0,   0,   128],  # 4  boat
    [128, 0,   128],  # 5  bottle
    [0,   128, 128],  # 6  bus
    [128, 128, 128],  # 7  car
    [64,  0,   0],    # 8  cat
    [192, 0,   0],    # 9  chair
    [64,  128, 0],    # 10 cow
    [192, 128, 0],    # 11 diningtable
    [64,  0,   128],  # 12 dog
    [192, 0,   128],  # 13 horse
    [64,  128, 128],  # 14 motorbike
    [192, 128, 128],  # 15 person
    [0,   64,  0],    # 16 pottedplant
    [128, 64,  0],    # 17 sheep
    [0,   192, 0],    # 18 sofa
    [128, 192, 0],    # 19 train
    [0,   64,  128],  # 20 tvmonitor
], dtype=np.uint8)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Segmentation result for a single frame."""
    video_id: str
    frame_idx: int
    timestamp_sec: float
    mask: np.ndarray                        # H×W uint8, class indices
    class_pixel_counts: dict[int, int] = field(default_factory=dict)

    def compute_pixel_counts(self) -> None:
        """Populate ``class_pixel_counts`` from the mask."""
        unique, counts = np.unique(self.mask, return_counts=True)
        self.class_pixel_counts = {
            int(cls): int(cnt) for cls, cnt in zip(unique, counts)
        }

    def metadata_dict(self, mask_relpath: str) -> dict:
        """Return a JSON-serialisable metadata dict."""
        if not self.class_pixel_counts:
            self.compute_pixel_counts()
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "mask_file": mask_relpath,
            "class_pixel_counts": {
                str(k): v for k, v in self.class_pixel_counts.items()
            },
        }


# ── Segmentation pipeline ────────────────────────────────────────────

class SegmentationPipeline:
    """DeepLabV3-based anatomy segmentation pipeline.

    Parameters
    ----------
    model_name : str
        One of ``"deeplabv3_resnet101"``, ``"deeplabv3_resnet50"``, or
        ``"deeplabv3_mobilenet_v3_large"``.
    model_weights : str | Path | None
        Path to a custom ``.pt`` state-dict checkpoint.  When ``None``
        the default COCO-pretrained weights are used.
    device : str | None
        PyTorch device.  ``None`` auto-selects CUDA if available.
    half : bool
        Use FP16 inference for higher throughput on supported GPUs.
    img_size : int
        Shorter side of the input is resized to this before inference.
        The aspect ratio is preserved.  Larger values give finer masks
        at the cost of speed.
    batch_size : int
        Number of frames per forward pass.
    class_names : list[str] | None
        Human-readable class names.  Defaults to COCO-VOC 21 classes.
    """

    def __init__(
        self,
        model_name: str = "deeplabv3_resnet101",
        model_weights: str | Path | None = None,
        device: str | None = None,
        half: bool = True,
        img_size: int = 512,
        batch_size: int = 8,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.img_size = img_size
        self.half = half
        self.batch_size = batch_size
        self.class_names = class_names or COCO_VOC_CLASSES

        # ── device selection ──────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.half and self.device == "cpu":
            log.warning("FP16 not supported on CPU – falling back to FP32.")
            self.half = False

        # ── model loading ─────────────────────────────────────────────
        log.info(
            "Loading segmentation model: %s  (device=%s, half=%s)",
            model_name, self.device, self.half,
        )
        self.model = self._load_model(model_name, model_weights)
        self.model.to(self.device)
        self.model.eval()
        if self.half:
            self.model.half()

        num_classes = self._probe_num_classes()
        log.info(
            "Model ready – %d classes, img_size=%d, batch_size=%d",
            num_classes, self.img_size, self.batch_size,
        )

    # ── model construction ────────────────────────────────────────────

    @staticmethod
    def _load_model(
        model_name: str,
        model_weights: str | Path | None,
    ) -> torch.nn.Module:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(_MODEL_REGISTRY)}"
            )
        constructor, default_weights = _MODEL_REGISTRY[model_name]

        if model_weights is not None:
            # Custom checkpoint: build model without pretrained weights,
            # then load the state dict.
            model = constructor(weights=None)
            state = torch.load(str(model_weights), map_location="cpu")
            # Support both raw state-dicts and {"model_state_dict": …}
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            log.info("Loaded custom weights from %s", model_weights)
        else:
            model = constructor(weights=default_weights)
            log.info("Loaded default pretrained weights (%s)", default_weights)

        return model

    def _probe_num_classes(self) -> int:
        """Determine the number of output classes from the model head."""
        # DeepLabV3 models expose classifier[-1] as the final conv layer
        head = self.model.classifier[-1]
        return head.out_channels

    # ── preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, bgr_frames: list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR numpy arrays to a normalised batch tensor.

        Steps:
        1. BGR → RGB
        2. Resize shorter side to ``self.img_size`` preserving aspect ratio
        3. Normalise with ImageNet mean/std
        4. Stack into (N, 3, H, W) tensor
        """
        tensors: list[torch.Tensor] = []
        for bgr in bgr_frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Resize so the shorter side == img_size
            h, w = rgb.shape[:2]
            if h < w:
                new_h = self.img_size
                new_w = int(w * self.img_size / h)
            else:
                new_w = self.img_size
                new_h = int(h * self.img_size / w)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            t = TF.to_tensor(rgb)  # (3, H, W), float32 [0, 1]
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            tensors.append(t)

        # Pad to uniform size within the batch (needed for torch.stack)
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded: list[torch.Tensor] = []
        for t in tensors:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            if pad_h > 0 or pad_w > 0:
                # F.pad order: (left, right, top, bottom)
                t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)
            padded.append(t)

        batch = torch.stack(padded, dim=0)
        batch = batch.to(self.device)
        if self.half:
            batch = batch.half()
        return batch

    # ── single-batch inference ────────────────────────────────────────

    @torch.inference_mode()
    def _predict_batch(
        self,
        bgr_frames: list[np.ndarray],
        original_sizes: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """Run segmentation on a batch of BGR frames.

        Returns a list of uint8 masks, each resized back to the
        original frame dimensions (H, W).
        """
        batch = self._preprocess(bgr_frames)
        output = self.model(batch)["out"]  # (N, C, H', W')
        preds = output.argmax(dim=1)       # (N, H', W')

        masks: list[np.ndarray] = []
        for i, (orig_h, orig_w) in enumerate(original_sizes):
            mask = preds[i].byte().cpu().numpy()  # uint8
            # Resize back to original frame resolution (nearest to keep
            # class indices intact)
            if (mask.shape[0], mask.shape[1]) != (orig_h, orig_w):
                mask = cv2.resize(
                    mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append(mask)

        return masks

    # ── full-video inference ──────────────────────────────────────────

    def process_video(
        self,
        video_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> Iterator[SegmentationResult]:
        """Run segmentation on every (strided) frame of a video.

        Yields :class:`SegmentationResult` objects in frame order.
        """
        video_path = Path(video_path)
        if video_id is None:
            video_id = video_path.stem

        with VideoReader(video_path, stride=stride) as reader:
            log.info(
                "Segmenting %s  (%dx%d, %.1f fps, ~%d frames, stride=%d)",
                video_id, reader.width, reader.height,
                reader.fps, reader.frame_count, stride,
            )
            t0 = time.perf_counter()
            frames_processed = 0

            for batch in reader.batched_frames(self.batch_size):
                indices, timestamps, bgr_frames = zip(*batch)
                original_sizes = [(f.shape[0], f.shape[1]) for f in bgr_frames]
                masks = self._predict_batch(list(bgr_frames), original_sizes)

                for idx, ts, mask in zip(indices, timestamps, masks):
                    frames_processed += 1
                    result = SegmentationResult(
                        video_id=video_id,
                        frame_idx=idx,
                        timestamp_sec=ts,
                        mask=mask,
                    )
                    result.compute_pixel_counts()
                    yield result

            elapsed = time.perf_counter() - t0
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            log.info(
                "Finished %s – %d frames in %.1fs  (%.1f FPS)",
                video_id, frames_processed, elapsed, fps_actual,
            )

    # ── convenience: save masks + metadata ────────────────────────────

    def process_video_to_masks(
        self,
        video_path: Path,
        output_dir: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> dict:
        """Process a video, save masks as PNGs and metadata as JSONL.

        Directory layout::

            output_dir/
                video01/
                    000000.png      ← class-index mask (uint8)
                    000025.png
                    …
                video01_masks.jsonl ← per-frame metadata

        Returns a summary dict.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        vid = video_id or video_path.stem
        mask_dir = output_dir / vid
        mask_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{vid}_masks.jsonl"

        total_frames = 0
        t0 = time.perf_counter()

        with open(jsonl_path, "w") as fh:
            for result in self.process_video(video_path, vid, stride):
                # Save mask as class-index PNG
                mask_filename = f"{result.frame_idx:06d}.png"
                mask_path = mask_dir / mask_filename
                cv2.imwrite(str(mask_path), result.mask)

                # Write metadata line
                mask_relpath = f"{vid}/{mask_filename}"
                meta = result.metadata_dict(mask_relpath)
                fh.write(json.dumps(meta) + "\n")

                total_frames += 1

        elapsed = time.perf_counter() - t0
        summary = {
            "video_id": vid,
            "total_frames": total_frames,
            "mask_dir": str(mask_dir),
            "metadata_file": str(jsonl_path),
            "elapsed_sec": round(elapsed, 2),
            "throughput_fps": round(total_frames / elapsed, 1) if elapsed > 0 else 0,
        }
        log.info("Saved %s  (%s)", mask_dir, summary)
        return summary
