"""Surgical phase recognition pipeline for Cholec80.

Classifies each video frame into one of the 7 standard Cholec80 surgical
phases using a ResNet-50 backbone (ImageNet-pretrained) with a 7-class
head.  Temporal smoothing via a sliding median filter and minimum-duration
enforcement produces stable, non-flickering phase labels suitable for
downstream workflow analysis.

Approach
--------
1. **Per-frame classification** — A ResNet-50 extracts a 2048-d feature
   vector per frame, projected through a linear head to 7 phase logits.
   With a fine-tuned checkpoint this achieves ~85-90 % frame accuracy on
   Cholec80.  Without fine-tuning (ImageNet features + random head) the
   raw predictions are poor, but the pipeline still runs end-to-end for
   integration testing; pass ``--model-weights`` with a trained ``.pt``
   checkpoint for real use.

2. **Temporal smoothing** — Raw per-frame predictions are noisy.  Two
   post-processing stages stabilise them:
   a. **Sliding median filter** (default window = 25 frames ~ 1 s at
      25 fps) replaces each prediction with the local mode/median.
   b. **Minimum-duration enforcement** (default = 250 frames ~ 10 s)
      removes phase segments shorter than the threshold by merging them
      into adjacent segments.

3. **Escalation** — After evaluation against ground-truth annotations,
   if overall accuracy is below the configurable threshold (default 70 %)
   a warning is emitted and the summary includes ``"escalate": true``.

Output format (JSON Lines)::

    {
      "video_id": "video01",
      "frame_idx": 0,
      "timestamp_sec": 0.0,
      "phase_id": 3,
      "phase_name": "GallbladderDissection",
      "confidence": 0.87,
      "raw_phase_id": 3,
      "smoothed": true
    }
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as TF

from src.video import VideoReader

log = logging.getLogger(__name__)

# ── Cholec80 phase taxonomy ──────────────────────────────────────────

CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

PHASE_TO_ID = {name: idx for idx, name in enumerate(CHOLEC80_PHASES)}
ID_TO_PHASE = {idx: name for idx, name in enumerate(CHOLEC80_PHASES)}
NUM_PHASES = len(CHOLEC80_PHASES)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """Phase classification result for a single frame."""
    video_id: str
    frame_idx: int
    timestamp_sec: float
    phase_id: int
    phase_name: str
    confidence: float
    raw_phase_id: int
    smoothed: bool

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "confidence": round(self.confidence, 4),
            "raw_phase_id": self.raw_phase_id,
            "smoothed": self.smoothed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Model construction ───────────────────────────────────────────────

def build_phase_model(
    num_classes: int = NUM_PHASES,
    weights_path: str | Path | None = None,
) -> nn.Module:
    """Build a ResNet-50 with a ``num_classes``-way head.

    If *weights_path* is ``None`` the backbone uses ImageNet weights and
    the head is randomly initialised (useful for integration testing
    only — pass a fine-tuned checkpoint for real inference).
    """
    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    if weights_path is not None:
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict(state)
        log.info("Loaded phase-recognition weights from %s", weights_path)
    else:
        log.info(
            "No --model-weights supplied; using ImageNet backbone + "
            "random head (integration-test mode — predictions will be "
            "unreliable until a fine-tuned checkpoint is provided)."
        )

    return backbone


# ── Temporal smoothing ────────────────────────────────────────────────

def median_smooth(labels: list[int], window: int) -> list[int]:
    """Apply a sliding-window majority-vote (mode) filter.

    For each position the most common label within the window centred on
    that position is chosen.  This is more appropriate than a numerical
    median for categorical phase IDs.
    """
    if window <= 1:
        return list(labels)
    half = window // 2
    n = len(labels)
    smoothed: list[int] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counter = Counter(labels[lo:hi])
        smoothed.append(counter.most_common(1)[0][0])
    return smoothed


def enforce_min_duration(labels: list[int], min_len: int) -> list[int]:
    """Merge segments shorter than *min_len* frames into neighbours.

    Scans left-to-right.  Any segment shorter than *min_len* is replaced
    by the label of the preceding segment (or the following segment if
    it is the first).
    """
    if min_len <= 1 or not labels:
        return list(labels)

    # Build run-length encoding
    runs: list[tuple[int, int]] = []  # (label, length)
    cur_label = labels[0]
    cur_len = 1
    for lbl in labels[1:]:
        if lbl == cur_label:
            cur_len += 1
        else:
            runs.append((cur_label, cur_len))
            cur_label = lbl
            cur_len = 1
    runs.append((cur_label, cur_len))

    # Merge short runs into the previous segment
    merged: list[tuple[int, int]] = []
    for label, length in runs:
        if length < min_len and merged:
            prev_label, prev_len = merged[-1]
            merged[-1] = (prev_label, prev_len + length)
        else:
            merged.append((label, length))

    # Expand back to per-frame labels
    result: list[int] = []
    for label, length in merged:
        result.extend([label] * length)
    return result


# ── Ground-truth annotation loader ───────────────────────────────────

def load_phase_annotations(annotation_path: Path) -> dict[int, int]:
    """Load Cholec80 ``videoXX-phase.txt`` into a {frame_idx: phase_id} map.

    The file format is tab-separated: ``frame_idx<TAB>PhaseName``.
    The first line is a header row.
    """
    mapping: dict[int, int] = {}
    with open(annotation_path) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            frame_idx = int(parts[0])
            phase_name = parts[1].strip()
            phase_id = PHASE_TO_ID.get(phase_name, -1)
            if phase_id >= 0:
                mapping[frame_idx] = phase_id
    return mapping


def evaluate_predictions(
    predictions: dict[int, int],
    ground_truth: dict[int, int],
) -> dict:
    """Compare predicted phase IDs against ground truth.

    Returns a dict with overall accuracy, per-phase precision, and
    per-phase recall.  Only frames present in both dicts are evaluated.
    """
    common_frames = sorted(set(predictions) & set(ground_truth))
    if not common_frames:
        return {"num_evaluated": 0, "accuracy": 0.0, "per_phase": {}}

    correct = 0
    per_phase_tp: dict[int, int] = {}
    per_phase_fp: dict[int, int] = {}
    per_phase_fn: dict[int, int] = {}
    for pid in range(NUM_PHASES):
        per_phase_tp[pid] = 0
        per_phase_fp[pid] = 0
        per_phase_fn[pid] = 0

    for fidx in common_frames:
        pred = predictions[fidx]
        gt = ground_truth[fidx]
        if pred == gt:
            correct += 1
            per_phase_tp[gt] += 1
        else:
            per_phase_fp[pred] = per_phase_fp.get(pred, 0) + 1
            per_phase_fn[gt] += 1

    accuracy = correct / len(common_frames)

    per_phase: dict[str, dict] = {}
    for pid in range(NUM_PHASES):
        tp = per_phase_tp[pid]
        fp = per_phase_fp.get(pid, 0)
        fn = per_phase_fn[pid]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_phase[CHOLEC80_PHASES[pid]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return {
        "num_evaluated": len(common_frames),
        "accuracy": round(accuracy, 4),
        "per_phase": per_phase,
    }


# ── Phase recognition pipeline ───────────────────────────────────────

class PhaseRecognitionPipeline:
    """Surgical phase recognition with temporal smoothing.

    Parameters
    ----------
    model_weights : str | Path | None
        Path to a fine-tuned ``.pt`` checkpoint.  ``None`` uses ImageNet
        backbone + random head (integration-test mode).
    device : str | None
        PyTorch device.  ``None`` auto-selects CUDA if available.
    half : bool
        Use FP16 inference on supported GPUs.
    img_size : int
        Input image size (centre-cropped square).
    batch_size : int
        Frames per forward pass.
    smooth_window : int
        Majority-vote sliding window size (frames).  0 disables.
    min_phase_duration : int
        Minimum segment length (frames) for duration enforcement.
        0 disables.
    accuracy_threshold : float
        Minimum acceptable accuracy.  If evaluation falls below this,
        the summary includes ``"escalate": true`` and a warning is
        logged.
    """

    def __init__(
        self,
        model_weights: str | Path | None = None,
        device: str | None = None,
        half: bool = True,
        img_size: int = 224,
        batch_size: int = 32,
        smooth_window: int = 25,
        min_phase_duration: int = 250,
        accuracy_threshold: float = 0.70,
    ) -> None:
        self.img_size = img_size
        self.half = half
        self.batch_size = batch_size
        self.smooth_window = smooth_window
        self.min_phase_duration = min_phase_duration
        self.accuracy_threshold = accuracy_threshold

        # ── device ────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.half and self.device == "cpu":
            log.warning("FP16 not supported on CPU – falling back to FP32.")
            self.half = False

        # ── model ─────────────────────────────────────────────────────
        log.info(
            "Loading phase model  (device=%s, half=%s, weights=%s)",
            self.device, self.half, model_weights or "none (integration-test)",
        )
        self.model = build_phase_model(
            num_classes=NUM_PHASES,
            weights_path=model_weights,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.half:
            self.model.half()
        log.info(
            "Phase model ready – %d classes, batch_size=%d, "
            "smooth_window=%d, min_duration=%d",
            NUM_PHASES, self.batch_size,
            self.smooth_window, self.min_phase_duration,
        )

    # ── preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, bgr_frames: list[np.ndarray]) -> torch.Tensor:
        """BGR frames -> normalised (N, 3, 224, 224) tensor."""
        tensors: list[torch.Tensor] = []
        for bgr in bgr_frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Resize then centre-crop to img_size x img_size
            h, w = rgb.shape[:2]
            scale = (self.img_size + 32) / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Centre crop
            y0 = (new_h - self.img_size) // 2
            x0 = (new_w - self.img_size) // 2
            rgb = rgb[y0:y0 + self.img_size, x0:x0 + self.img_size]

            t = TF.to_tensor(rgb)
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        if self.half:
            batch = batch.half()
        return batch

    # ── batch inference ───────────────────────────────────────────────

    @torch.inference_mode()
    def _predict_batch(
        self, bgr_frames: list[np.ndarray]
    ) -> tuple[list[int], list[float]]:
        """Classify a batch of frames.

        Returns ``(phase_ids, confidences)`` lists.
        """
        batch = self._preprocess(bgr_frames)
        logits = self.model(batch)                  # (N, 7)
        probs = torch.softmax(logits.float(), dim=1)
        confs, preds = probs.max(dim=1)
        return preds.cpu().tolist(), confs.cpu().tolist()

    # ── full-video inference ──────────────────────────────────────────

    def process_video(
        self,
        video_path: Path,
        video_id: str | None = None,
        stride: int = 1,
    ) -> list[PhaseResult]:
        """Run phase recognition on a video.

        Returns a list of :class:`PhaseResult` (one per processed
        frame), with temporal smoothing applied.
        """
        video_path = Path(video_path)
        if video_id is None:
            video_id = video_path.stem

        # Pass 1: per-frame classification
        raw_indices: list[int] = []
        raw_timestamps: list[float] = []
        raw_frame_idxs: list[int] = []
        raw_confidences: list[float] = []

        with VideoReader(video_path, stride=stride) as reader:
            log.info(
                "Phase-classifying %s  (%dx%d, %.1f fps, ~%d frames, stride=%d)",
                video_id, reader.width, reader.height,
                reader.fps, reader.frame_count, stride,
            )
            t0 = time.perf_counter()

            for batch_items in reader.batched_frames(self.batch_size):
                indices, timestamps, bgr_frames = zip(*batch_items)
                phase_ids, confs = self._predict_batch(list(bgr_frames))

                raw_frame_idxs.extend(indices)
                raw_timestamps.extend(timestamps)
                raw_indices.extend(phase_ids)
                raw_confidences.extend(confs)

        elapsed_classify = time.perf_counter() - t0
        n = len(raw_indices)
        log.info(
            "Classified %d frames in %.1fs (%.1f FPS)",
            n, elapsed_classify,
            n / elapsed_classify if elapsed_classify > 0 else 0,
        )

        # Pass 2: temporal smoothing
        smoothed_ids = list(raw_indices)
        applied_smoothing = False

        if self.smooth_window > 1 and n > 1:
            smoothed_ids = median_smooth(smoothed_ids, self.smooth_window)
            applied_smoothing = True

        if self.min_phase_duration > 1 and n > 1:
            smoothed_ids = enforce_min_duration(
                smoothed_ids, self.min_phase_duration
            )
            applied_smoothing = True

        # Build result objects
        results: list[PhaseResult] = []
        for i in range(n):
            sid = smoothed_ids[i]
            results.append(PhaseResult(
                video_id=video_id,
                frame_idx=raw_frame_idxs[i],
                timestamp_sec=raw_timestamps[i],
                phase_id=sid,
                phase_name=ID_TO_PHASE.get(sid, f"phase_{sid}"),
                confidence=raw_confidences[i],
                raw_phase_id=raw_indices[i],
                smoothed=applied_smoothing and (sid != raw_indices[i]),
            ))

        return results

    # ── convenience: save + optional evaluate ─────────────────────────

    def process_video_to_jsonl(
        self,
        video_path: Path,
        output_path: Path,
        video_id: str | None = None,
        stride: int = 1,
        annotation_path: Path | None = None,
    ) -> dict:
        """Classify, smooth, save to JSONL, and optionally evaluate.

        Parameters
        ----------
        annotation_path : Path | None
            Path to ``videoXX-phase.txt`` ground-truth file.  If provided
            the summary includes accuracy metrics and escalation status.

        Returns a summary dict.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vid = video_id or video_path.stem

        t0 = time.perf_counter()
        results = self.process_video(video_path, vid, stride)

        with open(output_path, "w") as fh:
            for r in results:
                fh.write(r.to_json() + "\n")

        elapsed = time.perf_counter() - t0
        summary: dict = {
            "video_id": vid,
            "total_frames": len(results),
            "elapsed_sec": round(elapsed, 2),
            "throughput_fps": round(len(results) / elapsed, 1) if elapsed > 0 else 0,
        }

        # Optional evaluation against ground truth
        if annotation_path is not None:
            annotation_path = Path(annotation_path)
            if annotation_path.exists():
                gt = load_phase_annotations(annotation_path)
                preds = {r.frame_idx: r.phase_id for r in results}
                eval_result = evaluate_predictions(preds, gt)
                summary["evaluation"] = eval_result

                acc = eval_result["accuracy"]
                escalate = acc < self.accuracy_threshold
                summary["escalate"] = escalate

                if escalate:
                    log.warning(
                        "ESCALATE: %s accuracy %.1f%% is below threshold "
                        "%.1f%%. Consider fine-tuning the model or adjusting "
                        "smoothing parameters.",
                        vid, acc * 100, self.accuracy_threshold * 100,
                    )
                else:
                    log.info(
                        "%s accuracy: %.1f%% (threshold: %.1f%%)",
                        vid, acc * 100, self.accuracy_threshold * 100,
                    )
            else:
                log.warning(
                    "Annotation file not found: %s – skipping evaluation.",
                    annotation_path,
                )

        log.info("Saved %s  (%s)", output_path, summary)
        return summary
