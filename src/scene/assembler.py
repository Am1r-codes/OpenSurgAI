"""Unified surgical scene representation assembler.

Consumes the JSONL / metadata outputs from the three upstream pipelines
(detection, segmentation, phase recognition) and joins them by
``frame_idx`` into a single per-frame **SceneState** JSON object.  The
assembled output is the sole input to downstream explanation modules.

Input files consumed
--------------------
1. **Detection** — ``videoXX_detections.jsonl`` (from ``run_detection.py``)
   Each line: ``{video_id, frame_idx, timestamp_sec, detections: [{bbox, class_id, class_name, confidence}]}``

2. **Segmentation** — ``videoXX_masks.jsonl`` (from ``run_segmentation.py``)
   Each line: ``{video_id, frame_idx, timestamp_sec, mask_file, class_pixel_counts: {"0": N, ...}}``

3. **Phase** — ``videoXX_phases.jsonl`` (from ``run_phase_recognition.py``)
   Each line: ``{video_id, frame_idx, timestamp_sec, phase_id, phase_name, confidence, raw_phase_id, smoothed}``

Unified output schema (JSON Lines)
-----------------------------------
::

    {
      "video_id": "video01",
      "frame_idx": 0,
      "timestamp_sec": 0.0,

      "phase": {
        "phase_id": 3,
        "phase_name": "GallbladderDissection",
        "confidence": 0.87,
        "raw_phase_id": 3,
        "smoothed": false
      },

      "instruments": [
        {
          "bbox": [120.5, 80.3, 310.2, 250.1],
          "class_id": 0,
          "class_name": "Grasper",
          "confidence": 0.93
        }
      ],
      "instrument_count": 1,

      "anatomy": {
        "mask_file": "video01/000000.png",
        "class_pixel_counts": {"0": 245120, "15": 62480},
        "classes_present": ["background", "person"]
      },

      "sources": {
        "has_detection": true,
        "has_segmentation": true,
        "has_phase": true
      }
    }

Design choices
--------------
- **Graceful degradation**: any of the three upstream sources may be
  absent for a given frame (or entirely missing for the video).  The
  ``sources`` dict tells downstream consumers which data is available.
- **Frame alignment**: the assembler collects the union of all
  ``frame_idx`` values across sources and produces one SceneState per
  frame.  Missing data is filled with sensible defaults (empty
  instrument list, null phase, null anatomy).
- **Deterministic ordering**: output is sorted by ``frame_idx``.
- **No mask pixels**: the assembler references mask files by path but
  does *not* embed raw pixel data — keeping the JSONL lightweight.
  Downstream modules that need the actual mask load it from disk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── JSONL loaders ─────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file into a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("%s:%d – skipping malformed line: %s", path, lineno, exc)
    return records


def _index_by_frame(records: list[dict]) -> dict[int, dict]:
    """Index a list of per-frame dicts by ``frame_idx``."""
    index: dict[int, dict] = {}
    for rec in records:
        fidx = rec.get("frame_idx")
        if fidx is not None:
            index[int(fidx)] = rec
    return index


# ── Class-name resolver for segmentation pixel counts ────────────────

# Segmentation class_pixel_counts use string class-id keys.  We resolve
# them to human-readable names when a class-name list is provided.

def _resolve_classes_present(
    pixel_counts: dict[str, int],
    class_names: list[str] | None,
) -> list[str]:
    """Return sorted list of non-background class names present."""
    present: list[str] = []
    for cls_str, count in pixel_counts.items():
        cls_id = int(cls_str)
        if cls_id == 0 or count == 0:
            continue
        if class_names and cls_id < len(class_names):
            present.append(class_names[cls_id])
        else:
            present.append(f"class_{cls_id}")
    return sorted(present)


# ── SceneState data class ────────────────────────────────────────────

@dataclass
class SceneState:
    """Unified per-frame surgical scene representation.

    This is the canonical structure consumed by downstream explanation
    modules.  Every field is always present; missing upstream data is
    represented by ``None`` / empty collections so consumers do not
    need to handle ``KeyError``.
    """
    video_id: str
    frame_idx: int
    timestamp_sec: float

    # Phase recognition
    phase: dict | None = None

    # Instrument detection
    instruments: list[dict] = field(default_factory=list)
    instrument_count: int = 0

    # Anatomy segmentation
    anatomy: dict | None = None

    # Provenance: which upstream sources contributed
    has_detection: bool = False
    has_segmentation: bool = False
    has_phase: bool = False

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "phase": self.phase,
            "instruments": self.instruments,
            "instrument_count": self.instrument_count,
            "anatomy": self.anatomy,
            "sources": {
                "has_detection": self.has_detection,
                "has_segmentation": self.has_segmentation,
                "has_phase": self.has_phase,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Scene assembler ──────────────────────────────────────────────────

class SceneAssembler:
    """Combines detection, segmentation, and phase outputs into unified
    per-frame SceneState objects.

    Parameters
    ----------
    segmentation_class_names : list[str] | None
        Human-readable class names for the segmentation model output.
        Index 0 is background.  If ``None``, class IDs are used as-is
        in the ``classes_present`` list.
    """

    def __init__(
        self,
        segmentation_class_names: list[str] | None = None,
    ) -> None:
        self.seg_class_names = segmentation_class_names

    # ── core assembly ─────────────────────────────────────────────────

    def assemble_video(
        self,
        video_id: str,
        detection_path: Path | None = None,
        segmentation_path: Path | None = None,
        phase_path: Path | None = None,
    ) -> list[SceneState]:
        """Load upstream JSONL files and assemble unified scene states.

        Any of the three paths may be ``None`` — the assembler produces
        output for the union of frame indices found across all provided
        sources.

        Returns a list of :class:`SceneState` sorted by ``frame_idx``.
        """
        # Load available sources
        det_index: dict[int, dict] = {}
        seg_index: dict[int, dict] = {}
        phase_index: dict[int, dict] = {}

        if detection_path and Path(detection_path).exists():
            det_records = _load_jsonl(Path(detection_path))
            det_index = _index_by_frame(det_records)
            log.info("Loaded %d detection frames from %s", len(det_index), detection_path)

        if segmentation_path and Path(segmentation_path).exists():
            seg_records = _load_jsonl(Path(segmentation_path))
            seg_index = _index_by_frame(seg_records)
            log.info("Loaded %d segmentation frames from %s", len(seg_index), segmentation_path)

        if phase_path and Path(phase_path).exists():
            phase_records = _load_jsonl(Path(phase_path))
            phase_index = _index_by_frame(phase_records)
            log.info("Loaded %d phase frames from %s", len(phase_index), phase_path)

        # Union of all frame indices
        all_frames = sorted(
            set(det_index) | set(seg_index) | set(phase_index)
        )
        if not all_frames:
            log.warning("No frames found for %s – all sources empty or missing.", video_id)
            return []

        log.info(
            "Assembling %d frames for %s  (det=%d, seg=%d, phase=%d)",
            len(all_frames), video_id,
            len(det_index), len(seg_index), len(phase_index),
        )

        # Determine best timestamp source (prefer phase > detection > segmentation)
        def _get_timestamp(fidx: int) -> float:
            for idx in (phase_index, det_index, seg_index):
                rec = idx.get(fidx)
                if rec and "timestamp_sec" in rec:
                    return float(rec["timestamp_sec"])
            return 0.0

        # Assemble
        scenes: list[SceneState] = []
        for fidx in all_frames:
            ts = _get_timestamp(fidx)
            state = SceneState(
                video_id=video_id,
                frame_idx=fidx,
                timestamp_sec=ts,
            )

            # ── detection ─────────────────────────────────────────────
            det_rec = det_index.get(fidx)
            if det_rec is not None:
                state.has_detection = True
                state.instruments = det_rec.get("detections", [])
                state.instrument_count = len(state.instruments)

            # ── segmentation ──────────────────────────────────────────
            seg_rec = seg_index.get(fidx)
            if seg_rec is not None:
                state.has_segmentation = True
                pixel_counts = seg_rec.get("class_pixel_counts", {})
                classes_present = _resolve_classes_present(
                    pixel_counts, self.seg_class_names
                )
                state.anatomy = {
                    "mask_file": seg_rec.get("mask_file"),
                    "class_pixel_counts": pixel_counts,
                    "classes_present": classes_present,
                }

            # ── phase ─────────────────────────────────────────────────
            phase_rec = phase_index.get(fidx)
            if phase_rec is not None:
                state.has_phase = True
                state.phase = {
                    "phase_id": phase_rec.get("phase_id"),
                    "phase_name": phase_rec.get("phase_name"),
                    "confidence": phase_rec.get("confidence"),
                    "raw_phase_id": phase_rec.get("raw_phase_id"),
                    "smoothed": phase_rec.get("smoothed"),
                }

            scenes.append(state)

        return scenes

    # ── convenience: assemble and write ───────────────────────────────

    def assemble_video_to_jsonl(
        self,
        video_id: str,
        output_path: Path,
        detection_path: Path | None = None,
        segmentation_path: Path | None = None,
        phase_path: Path | None = None,
    ) -> dict:
        """Assemble scene states for a video and write to JSONL.

        Returns a summary dict.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        scenes = self.assemble_video(
            video_id=video_id,
            detection_path=detection_path,
            segmentation_path=segmentation_path,
            phase_path=phase_path,
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            for scene in scenes:
                fh.write(scene.to_json() + "\n")

        elapsed = time.perf_counter() - t0

        # Compute source coverage stats
        n = len(scenes)
        n_det = sum(1 for s in scenes if s.has_detection)
        n_seg = sum(1 for s in scenes if s.has_segmentation)
        n_phase = sum(1 for s in scenes if s.has_phase)
        n_complete = sum(
            1 for s in scenes
            if s.has_detection and s.has_segmentation and s.has_phase
        )
        total_instruments = sum(s.instrument_count for s in scenes)

        summary = {
            "video_id": video_id,
            "total_frames": n,
            "frames_with_detection": n_det,
            "frames_with_segmentation": n_seg,
            "frames_with_phase": n_phase,
            "frames_complete": n_complete,
            "completeness_pct": round(n_complete / n * 100, 1) if n > 0 else 0,
            "total_instruments_detected": total_instruments,
            "elapsed_sec": round(elapsed, 4),
        }
        log.info("Assembled %s  (%s)", output_path, summary)
        return summary
